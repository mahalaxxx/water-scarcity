import pandas as pd
import random

random.seed(42)

# Approximate locations (demo only, real village data later)
places = [
    ("Ramanathapuram", "Ramanathapuram", 9.37, 78.83),
    ("Rameswaram", "Rameswaram", 9.29, 79.31),
    ("Mandapam", "Mandapam", 9.28, 79.12),
    ("Paramakudi", "Paramakudi", 9.55, 78.59),
    ("Kamuthi", "Kamuthi", 9.41, 78.37),
    ("Mudukulathur", "Mudukulathur", 9.34, 78.51),
    ("Kadaladi", "Kadaladi", 9.15, 78.34),
    ("Tiruvadanai", "Tiruvadanai", 9.77, 78.93),
    ("Thondi", "Tiruvadanai", 9.74, 79.01),
    ("Devipattinam", "Ramanathapuram", 9.48, 78.90),
    ("Uchipuli", "Mandapam", 9.30, 79.00),
    ("Keelakarai", "Keelakarai", 9.23, 78.78),
    ("Sayalkudi", "Kadaladi", 9.17, 78.47),
    ("Nainarkoil", "Nainarkoil", 9.50, 78.72),
    ("R.S.Mangalam", "R.S.Mangalam", 9.55, 78.97),
]

rows = []
for name, block, lat, lon in places:
    score = round(random.uniform(0.05, 0.95), 2)
    if score < 0.33:
        level = "Low"
    elif score < 0.66:
        level = "Medium"
    else:
        level = "High"
    rows.append({
        "village": name,
        "block": block,
        "lat": lat + random.uniform(-0.02, 0.02),
        "lon": lon + random.uniform(-0.02, 0.02),
        "risk_level": level,
        "risk_score": score,
        "rainfall_deficit": random.randint(0, 60),
        "reason": "Sample data",
    })

pd.DataFrame(rows).to_csv("outputs/risk_predictions.csv", index=False)
print("Done: outputs/risk_predictions.csv created")