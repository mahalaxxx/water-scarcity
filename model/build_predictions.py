import pandas as pd
from sklearn.ensemble import RandomForestClassifier

b = pd.read_csv("data/blocks.csv", encoding="utf-8")
r = pd.read_csv("data/rainfall.csv", encoding="utf-8")
df = b.merge(r, on="block_en")


def norm(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0


# Groundwater risk (0 to 1): saline = unusable, else stage of extraction
df["gw_stage_pct"] = df["gw_stage_pct"].fillna(0)
df["gw_risk"] = (df["gw_stage_pct"] / 100).clip(0, 1)
df.loc[df["gw_saline"] >= 1, "gw_risk"] = 1.0
df.loc[(df["gw_saline"] > 0) & (df["gw_saline"] < 1), "gw_risk"] = df["gw_risk"].clip(lower=0.5)

df["risk_score"] = (
    0.4 * norm(df["rainfall_deficit_pct"])
    + 0.3 * (1 - norm(df["rain_90d_mm"]))
    + 0.3 * df["gw_risk"]
).round(2)

df["label"] = pd.qcut(
    df["risk_score"].rank(method="first"), 3, labels=["Low", "Medium", "High"]
)

features = ["rainfall_deficit_pct", "rain_90d_mm", "gw_risk"]
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(df[features], df["label"])

df["risk_level"] = model.predict(df[features])
df["confidence"] = model.predict_proba(df[features]).max(axis=1).round(2)
df["rainfall_deficit"] = df["rainfall_deficit_pct"]

out = df[["block_en", "block_ta", "panchayat_villages", "lat", "lon",
          "risk_level", "risk_score", "rainfall_deficit", "rain_90d_mm",
          "gw_stage_pct", "gw_saline", "confidence"]]
out.to_csv("outputs/risk_predictions.csv", index=False, encoding="utf-8")

print(out[["block_en", "risk_level", "risk_score", "gw_saline"]])
print("Done: outputs/risk_predictions.csv updated")