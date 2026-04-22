import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import uuid
import logging
import threading
import concurrent.futures
from collections import deque
from multiprocessing import freeze_support
from typing import List, Tuple, Optional

# ---------------------------------------------------------
# 1. Observability
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DistillationEngine")

# ---------------------------------------------------------
# 2. Safety & Sensors
# ---------------------------------------------------------
class RMDSensor:
    """
    Relative Mahalanobis Distance 계산기.
    update()는 Welford 평균과 Sherman-Morrison 랭크-1 역공분산 갱신을 수행.
    200 스텝마다 누적합에서 전체 재계산하여 드리프트를 리셋.
    """
    def __init__(self, feature_dim: int, min_samples: int = 50, reg_lambda: float = 1e-4):
        self.feature_dim = feature_dim
        self.min_samples = min_samples
        self.reg_lambda   = reg_lambda
        self.mu      = torch.zeros(feature_dim)
        self.cov_inv = torch.eye(feature_dim)
        self.n       = 0
        # 전체 재계산용 누적합 — 냉각 기간 이후에도 계속 유지
        self._cov_acc = torch.zeros(feature_dim, feature_dim)

    def compute(self, x: torch.Tensor) -> float:
        x_norm = F.normalize(x, p=2, dim=0)
        delta  = x_norm - self.mu
        return torch.sqrt(torch.dot(delta, self.cov_inv @ delta)).item()

    def update(self, x: torch.Tensor) -> None:
        """
        Welford 온라인 업데이트 후 공분산 역행렬 갱신.

        공분산 점화식:
            C_n = (n-2)/(n-1) * C_{n-1} + (1/n) * u*v^T
        여기서 u = x - μ_{n-1},  v = x - μ_n  (Welford 벡터)

        → Sherman-Morrison 적용 시 먼저 스케일링:
            B = (n-1)/(n-2) * C_{n-1}^{-1}
        그 후 랭크-1 업데이트:
            C_n^{-1} = B - (1/n)(Bu)(v^T B) / (1 + (1/n)v^T Bu)

        수치 안정성을 위해 200 스텝마다 누적합으로 전체 재계산 (O(d³) 비용은 무시 가능).
        """
        x_norm = F.normalize(x, p=2, dim=0)
        mu_old = self.mu.clone()
        self.n += 1
        self.mu = mu_old + (x_norm - mu_old) / self.n   # Welford 평균

        u = x_norm - mu_old   # pre-update delta
        v = x_norm - self.mu  # post-update delta

        # 누적합은 항상 유지 (주기적 전체 재계산에 사용)
        self._cov_acc += torch.outer(u, v)

        if self.n < self.min_samples:
            return

        # 전체 재계산 조건: 냉각 종료 시 OR 200 스텝 주기
        if self.n == self.min_samples or self.n % 200 == 0:
            C = self._cov_acc / max(self.n - 1, 1) + self.reg_lambda * torch.eye(self.feature_dim)
            self.cov_inv = torch.linalg.inv(C)
            return

        # Sherman-Morrison — Welford 스케일링 팩터 (n-2)/(n-1) 반영
        # B = (n-1)/(n-2) * C_{n-1}^{-1}  →  divide by scale
        scale = (self.n - 2) / (self.n - 1)
        B     = self.cov_inv / scale

        factor = 1.0 / self.n
        B_u    = B @ u
        denom  = 1.0 + factor * torch.dot(v, B_u)

        if denom.abs().item() < 1e-9:   # 근-특이 업데이트 스킵
            return

        self.cov_inv = B - factor * torch.outer(B_u, B @ v) / denom


# ---------------------------------------------------------
# TDA 헬퍼 — ProcessPoolExecutor pickle을 위해 모듈 레벨 정의
# ---------------------------------------------------------
def _compute_rips_wasserstein(
    z_np: np.ndarray,
    prev_diagram_np: Optional[np.ndarray],
    max_edge_length: float,
    homology_dims: List[int],
) -> Tuple[float, np.ndarray]:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import Wasserstein as WD

    vrp = VietorisRipsPersistence(
        homology_dimensions=homology_dims,
        max_edge_length=max_edge_length,
        n_jobs=1,
    )
    z_3d    = z_np[np.newaxis, :, :]       # (1, n_points, n_features)
    new_diag = vrp.fit_transform(z_3d)     # (1, n_pairs, 3)

    if prev_diagram_np is None:
        return 0.0, new_diag

    both = np.concatenate([prev_diagram_np, new_diag], axis=0)
    wd   = WD(order=1, enable_autodiff=False)
    wd.fit(both)
    dist_matrix = wd.transform(both)        # (2, 2)
    return float(dist_matrix[0, 1]), new_diag


class TDA_Monitor:
    """
    배치 단위 비동기 위상 붕괴 감지기.
    1-tick 래그 패턴: 미완료 구간에는 0.0이 아닌 마지막 실측값을 유지해
    '변화 없음'과 '모르겠다'를 구분.
    """
    def __init__(self, window_size: int = 10, max_edge_length: float = 2.0,
                 homology_dims: Optional[List[int]] = None, n_workers: int = 2):
        self.wasserstein_history = deque(maxlen=window_size)
        self.mu    = 0.0
        self.sigma = 1e-5
        self.max_edge_length  = max_edge_length
        self.homology_dims    = homology_dims or [0, 1]
        self.prev_diagram: Optional[np.ndarray] = None
        self.last_known_w_dist = 0.0              # 미완료 구간 대체값 (0.0 고정 아님)
        self._executor       = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
        self._pending_future: Optional[concurrent.futures.Future] = None

    def check_topological_spike(self, z_batch: torch.Tensor) -> Tuple[bool, float]:
        z_np = z_batch.detach().cpu().numpy()

        # 이전 future 결과 수집 — 성공 시 last_known_w_dist 갱신
        if self._pending_future is not None and self._pending_future.done():
            try:
                w_dist_result, new_diag = self._pending_future.result()
                self.prev_diagram      = new_diag
                self.last_known_w_dist = w_dist_result   # 실측값 보존
            except Exception as e:
                logger.error(f"TDA 연산 실패: {e}")
            self._pending_future = None

        # 새 연산 제출 (비블로킹)
        if self._pending_future is None:
            self._pending_future = self._executor.submit(
                _compute_rips_wasserstein,
                z_np,
                self.prev_diagram,
                self.max_edge_length,
                self.homology_dims,
            )

        # 미완료 구간: 마지막 실측값 사용 (거짓 안정 신호 방지)
        current_w_dist = self.last_known_w_dist

        spike_level = (current_w_dist - self.mu) / (self.sigma + 1e-8)
        is_collapse  = spike_level > 3.0

        self.wasserstein_history.append(current_w_dist)
        self.mu    = float(np.mean(self.wasserstein_history))
        self.sigma = float(np.std(self.wasserstein_history))

        return is_collapse, spike_level


class ControlBarrierFunction:
    """V^pi 기반 단계적 안전 구역 제어기"""
    def evaluate_zone(self, v_pi: float, theta_risk: float) -> Tuple[str, float]:
        d = v_pi - theta_risk
        if d < 0:        return "TERMINAL", 0.0
        elif d < 5.0:    return "RED",      0.1
        elif d < 15.0:   return "YELLOW",   0.5
        else:            return "GREEN",    1.0


# ---------------------------------------------------------
# 3. Policy-Value Network
# ---------------------------------------------------------
class PolicyValueNet(nn.Module):
    """
    공유 트렁크 → 정책 헤드(logits) + 가치 헤드(V^π).
    confidence는 호출 측에서 온도 스케일링 후 계산하므로
    forward()는 raw logits를 그대로 반환.
    """
    def __init__(self, state_dim: int = 128, action_dim: int = 5,
                 hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden1), nn.LayerNorm(hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2),   nn.LayerNorm(hidden2), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden2, action_dim)
        self.value_head  = nn.Linear(hidden2, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        features = self.trunk(state)
        logits   = self.policy_head(features)

        # confidence: 온도 없는 raw entropy (Arena에서 온도 적용 후 재계산)
        probs      = F.softmax(logits, dim=-1)
        H          = -torch.sum(probs * torch.log(probs + 1e-9))
        confidence = (1.0 - H / torch.log(torch.tensor(float(self.action_dim)))).item()

        v_pi = self.value_head(features).squeeze(-1).item()
        return logits, confidence, v_pi


# ---------------------------------------------------------
# 4. Agents
# ---------------------------------------------------------
class BiasedAgent:
    def __init__(self, agent_id: str, bias_type: str,
                 state_dim: int = 128, action_dim: int = 5):
        self.id        = agent_id
        self.bias_type = bias_type
        self.net       = PolicyValueNet(state_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        return self.net(state)


# ---------------------------------------------------------
# 5. 보상 함수 및 훈련 스켈레톤
# ---------------------------------------------------------
def reward_fn(
    state: torch.Tensor,
    action: int,
    next_state: torch.Tensor,
    bias_type: str,
) -> float:
    """
    State L2-norm 변화를 프록시로 사용하는 합성 보상.
    실 배포 시 PnL / Sharpe 기반 보상으로 교체 필요 (smoke test 용도).

    HighVol(킬 중심): 양의 전이 보상, 음의 전이 2배 패널티
    LowVol(오브젝트): 안정적 소폭 증가 보상, 변동성 패널티
    """
    delta = (next_state.norm() - state.norm()).item()
    if bias_type == "HighVol":
        return max(0.0, delta) - max(0.0, -delta) * 2.0
    elif bias_type == "LowVol":
        # 보상 스케일 조정: LowVol 최대값(≈0.05)이 너무 작아 gradient 소실 위험
        return (delta * 0.5 - abs(delta - 0.1)) * 5.0
    else:
        raise ValueError(f"Unknown bias_type: {bias_type}")


def train_agent_skeleton(
    agent: BiasedAgent,
    n_steps: int = 1000,
    lr: float     = 1e-3,
    gamma: float  = 0.99,
    state_dim: int = 128,
) -> None:
    """
    A2C 훈련 스켈레톤 (합성 전이 사용).
    bias_type별 보상 신호 차이로 V^π 분포가 다르게 수렴.
    실 환경에서는 state/next_state를 env.step()으로 교체.
    """
    optimizer = torch.optim.Adam(agent.net.parameters(), lr=lr)

    for step in range(n_steps):
        state  = torch.randn(state_dim)
        logits, _, v_pi = agent.forward(state)

        probs    = F.softmax(logits, dim=-1)
        dist_cat = torch.distributions.Categorical(probs)
        action   = dist_cat.sample()
        log_prob = dist_cat.log_prob(action)

        next_state = state + 0.1 * torch.randn(state_dim)
        r = reward_fn(state, action.item(), next_state, agent.bias_type)

        # v_next는 반드시 detach — TD 타겟 경로로 gradient 역전파 방지
        with torch.no_grad():
            _, _, v_next = agent.forward(next_state)
        td_target = r + gamma * v_next

        features    = agent.net.trunk(state)
        v_pi_tensor = agent.net.value_head(features).squeeze(-1)
        value_loss  = F.mse_loss(v_pi_tensor, torch.tensor(td_target))

        advantage   = td_target - v_pi
        policy_loss = -log_prob * advantage
        entropy     = -torch.sum(probs * torch.log(probs + 1e-9))
        loss        = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.net.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 200 == 0:
            logger.info(
                f"[{agent.id}] step={step} | v_pi={v_pi:.3f} | "
                f"reward={r:.3f} | loss={loss.item():.4f}"
            )


# ---------------------------------------------------------
# 6. Main Arena
# ---------------------------------------------------------
class Arena:
    def __init__(self, agents: List[BiasedAgent]):
        self.agents      = agents
        self.rmd_sensor  = RMDSensor(feature_dim=128)
        self.tda_monitor = TDA_Monitor()
        self.cbf         = ControlBarrierFunction()
        self.theta_risk  = -5.0
        self._halt_event = threading.Event()

    @property
    def halt_system(self) -> bool:
        return self._halt_event.is_set()

    @halt_system.setter
    def halt_system(self, value: bool) -> None:
        self._halt_event.set() if value else self._halt_event.clear()

    def tick_auction(self, state_features: torch.Tensor, corr_id: str):
        if self.halt_system:
            logger.error(f"[{corr_id}] SYSTEM HALTED.")
            return

        rmd_score   = self.rmd_sensor.compute(state_features)
        temperature = 1.0 + 0.5 * rmd_score
        self.rmd_sensor.update(state_features)
        logger.info(f"[{corr_id}] RMD: {rmd_score:.4f} | Temp: {temperature:.4f}")

        bids = []
        for agent in self.agents:
            logits, _, v_pi = agent.forward(state_features)

            zone, damping = self.cbf.evaluate_zone(v_pi, self.theta_risk)
            if zone == "TERMINAL":
                logger.warning(f"[{corr_id}] Agent {agent.id} ejected (TERMINAL).")
                continue

            # 온도를 logits에 먼저 적용한 뒤 entropy 기반 confidence 계산
            # (pre-application: 이상치 상태일수록 confidence가 더 많이 억제됨)
            probs_T = F.softmax(logits.detach() / temperature, dim=-1)
            H_T     = -torch.sum(probs_T * torch.log(probs_T + 1e-9))
            H_max   = torch.log(torch.tensor(float(logits.shape[-1])))
            conf    = (1.0 - H_T / H_max).item()

            scaled_conf = conf * damping
            bids.append((scaled_conf, agent, logits, zone))

        if not bids:
            logger.critical(f"[{corr_id}] All agents TERMINAL. System halt.")
            self.halt_system = True
            return

        bids.sort(key=lambda x: x[0], reverse=True)
        winning_conf, winning_agent, selected_action, zone = bids[0]
        logger.info(f"[{corr_id}] Winner: {winning_agent.id} | Conf: {winning_conf:.4f} | Zone: {zone}")
        # TODO: env.step(selected_action)

    def run_batch_monitor(self, latent_batch: torch.Tensor):
        corr_id = str(uuid.uuid4())[:8]
        is_collapse, spike = self.tda_monitor.check_topological_spike(latent_batch)
        if is_collapse:
            logger.critical(f"[{corr_id}] TOPOLOGICAL SPIKE (Z={spike:.2f}). Kill switch.")
            self.halt_system = True
        else:
            logger.info(f"[{corr_id}] Topology Stable (Z={spike:.2f}).")


# ---------------------------------------------------------
# 실행 시뮬레이션
# ---------------------------------------------------------
if __name__ == "__main__":
    freeze_support()   # Windows ProcessPoolExecutor 무한 재귀 방지

    agents = [
        BiasedAgent(agent_id="Faker_Aggro_01", bias_type="HighVol"),
        BiasedAgent(agent_id="Chovy_Safe_02",  bias_type="LowVol"),
    ]
    arena = Arena(agents)

    for tick in range(5):
        arena.tick_auction(torch.randn(128), str(uuid.uuid4())[:8])

    arena.run_batch_monitor(torch.randn(64, 32))
