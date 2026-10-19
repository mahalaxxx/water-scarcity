import pandas as pd
import requests
import time

df = pd.read_csv("data/blocks.csv", encoding="utf-8")
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

END = pd.Timestamp("2026-10-10")
START = END - pd.Timedelta(days=89)

rows = []
for _, r in df.iterrows():
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": round(r["lon"], 3),
        "latitude": round(r["lat"], 3),
        "start": "20160101",
        "end": "20261010",
        "format": "JSON",
    }
    print("Downloading:", r["block_en"])
    resp = requests.get(URL, params=params, timeout=120)
    data = resp.json()
    rain = pd.Series(data["properties"]["parameter"]["PRECTOTCORR"])
    rain.index = pd.to_datetime(rain.index, format="%Y%m%d")
    rain = rain.where(rain >= 0)

    recent = rain[START:END].sum()
    normals = []
    for y in range(2016, 2026):
        normals.append(rain[START.replace(year=y):END.replace(year=y)].sum())
    normal = sum(normals) / len(normals)
    deficit = max(0, (normal - recent) / normal * 100) if normal > 0 else 0

    rows.append({
        "block_en": r["block_en"],
        "rain_90d_mm": round(recent, 1),
        "normal_90d_mm": round(normal, 1),
        "rainfall_deficit_pct": round(deficit, 1),
    })
    time.sleep(1)

pd.DataFrame(rows).to_csv("data/rainfall.csv", index=False, encoding="utf-8")
print("Done: data/rainfall.csv created")