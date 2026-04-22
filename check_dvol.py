"""
Deribit DVOL Index 데이터 가용성 확인
"""
import requests
import pandas as pd
from datetime import datetime

BASE = "https://www.deribit.com/api/v2/public"

def get_dvol(currency="BTC", start_ts=None, end_ts=None, count=1000):
    params = {
        "currency":   currency,
        "resolution": "1D",
        "count":      count,
    }
    if start_ts: params["start_timestamp"] = start_ts
    if end_ts:   params["end_timestamp"]   = end_ts

    r = requests.get(f"{BASE}/get_volatility_index_data", params=params, timeout=15)
    data = r.json()
    if "result" not in data:
        print("API 오류:", data)
        return None
    rows = data["result"]["data"]  # [[ts, open, high, low, close], ...]
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("date")[["close"]].rename(columns={"close":"dvol"})

print("DVOL BTC 전체 범위 확인 (페이지 나눠서)...")
frames = []
# 2021부터 2026까지 1년 단위로 페이지
for year in range(2021, 2027):
    s = int(pd.Timestamp(f"{year}-01-01").timestamp() * 1000)
    e = int(pd.Timestamp(f"{year}-12-31").timestamp() * 1000)
    chunk = get_dvol("BTC", start_ts=s, end_ts=e, count=400)
    if chunk is not None and len(chunk) > 0:
        frames.append(chunk)
        print(f"  {year}: {len(chunk)}행, {chunk.index.min().date()} ~ {chunk.index.max().date()}")
    else:
        print(f"  {year}: 데이터 없음")

df = pd.concat(frames).drop_duplicates().sort_index() if frames else None
if df is not None:
    print(f"  rows: {len(df)}")
    print(f"  시작: {df.index.min().date()}")
    print(f"  끝:   {df.index.max().date()}")
    print(f"\n최초 5행:\n{df.head()}")
    print(f"\n크래시 주변 값:")
    for name, date in [("2018 Crash","2018-01-08"),
                        ("2020 COVID","2020-03-12"),
                        ("2022 FTX", "2022-11-08")]:
        try:
            idx = df.index.get_indexer([pd.Timestamp(date)], method='nearest')[0]
            val = df.iloc[idx]["dvol"]
            print(f"  {name} ({date}): DVOL={val:.1f}")
        except Exception as e:
            print(f"  {name}: 데이터 없음 ({e})")

    df.to_csv("dvol_btc.csv")
    print("\n저장: dvol_btc.csv")
