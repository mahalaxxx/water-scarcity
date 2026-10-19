import time
from concurrent.futures import ThreadPoolExecutor

import folium
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from streamlit_folium import st_folium

st.set_page_config(page_title="Water Scarcity - Ramanathapuram", layout="wide")

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
CACHE_HOURS = 6      # Ivvalavu neram oru thadava dhaan thirumba download aagum
LATENCY_DAYS = 7     # NASA POWER-ku konjam naal delay irukkum

COLORS = {"Low": "green", "Medium": "orange", "High": "red"}

TXT = {
    "en": {
        "title": "Water Scarcity Prediction - Ramanathapuram District",
        "note": "Data notes: Block-level view (11 blocks). Rainfall = NASA POWER "
                "(live download, ~50 km grid). Groundwater = CGWB 2024 (firka data "
                "matched to block by name, updated yearly). Risk = relative ranking "
                "among blocks (proxy score).",
        "loading": "Downloading latest rainfall from NASA POWER...",
        "refresh": "Refresh data now",
        "live_ok": "Live rainfall data up to",
        "fetched": "downloaded at",
        "live_fail": "Live download failed. Showing the last saved rainfall data.",
        "total": "Total blocks", "high": "High risk",
        "medium": "Medium risk", "low": "Low risk",
        "details": "Block details", "select": "Select block",
        "villages": "Panchayat villages", "risk_level": "Risk level",
        "risk_score": "Risk score", "deficit": "Rainfall deficit",
        "rain": "Rain (last 90 days)", "gw": "Groundwater (CGWB 2024)",
        "gw_saline": "Saline (quality problem)",
        "gw_partial": "Partly saline",
        "gw_safe": "Safe, extraction",
        "conf": "Model confidence",
        "reason": "Reason", "reco": "Recommended action",
        "alert": "Alert message preview", "table": "High risk blocks",
        "cols": ["Block", "Risk score", "Rainfall deficit %"],
        "levels": {"Low": "Low", "Medium": "Medium", "High": "High"},
    },
    "ta": {
        "title": "நீர் பற்றாக்குறை முன்னறிவிப்பு - இராமநாதபுரம் மாவட்டம்",
        "note": "தரவு குறிப்பு: ஒன்றிய அளவிலான பார்வை (11 ஒன்றியங்கள்). மழை அளவு = "
                "NASA POWER (நேரடி பதிவிறக்கம், ~50 கி.மீ கிரிட்). நிலத்தடி நீர் = "
                "CGWB 2024 (ஃபிர்கா தரவு, பெயர் அடிப்படையில் இணைப்பு, ஆண்டுதோறும் "
                "புதுப்பிக்கப்படும்). ஆபத்து = ஒன்றியங்களுக்கிடையே ஒப்பீட்டு தரவரிசை.",
        "loading": "NASA POWER-இலிருந்து சமீபத்திய மழை தரவு பெறப்படுகிறது...",
        "refresh": "தரவை இப்போது புதுப்பிக்கவும்",
        "live_ok": "நேரடி மழை தரவு, தேதி வரை",
        "fetched": "பெறப்பட்ட நேரம்",
        "live_fail": "நேரடி பதிவிறக்கம் தோல்வி. கடைசியாக சேமித்த மழை தரவு காட்டப்படுகிறது.",
        "total": "மொத்த ஒன்றியங்கள்", "high": "அதிக ஆபத்து",
        "medium": "நடுத்தர ஆபத்து", "low": "குறைந்த ஆபத்து",
        "details": "ஒன்றிய விவரங்கள்", "select": "ஒன்றியத்தைத் தேர்வு செய்க",
        "villages": "ஊராட்சி கிராமங்கள்", "risk_level": "ஆபத்து நிலை",
        "risk_score": "ஆபத்து மதிப்பெண்", "deficit": "மழை பற்றாக்குறை",
        "rain": "மழை (கடந்த 90 நாட்கள்)",
        "gw": "நிலத்தடி நீர் (CGWB 2024)",
        "gw_saline": "உவர்ப்பு (தரப் பிரச்சனை)",
        "gw_partial": "பகுதியளவு உவர்ப்பு",
        "gw_safe": "பாதுகாப்பான நிலை, உறிஞ்சுதல்",
        "conf": "மாடல் நம்பகத்தன்மை",
        "reason": "காரணம்", "reco": "பரிந்துரைக்கப்படும் நடவடிக்கை",
        "alert": "எச்சரிக்கை செய்தி முன்னோட்டம்",
        "table": "அதிக ஆபத்து ஒன்றியங்கள்",
        "cols": ["ஒன்றியம்", "ஆபத்து மதிப்பெண்", "மழை பற்றாக்குறை %"],
        "levels": {"Low": "குறைவு", "Medium": "நடுத்தரம்", "High": "அதிகம்"},
    },
}

RECO = {
    "High": {
        "en": "Send water tankers first. Check borewells and desilt nearby tanks. "
              "Inform the block development officer.",
        "ta": "முதலில் தண்ணீர் லாரிகளை அனுப்பவும். ஆழ்துளை கிணறுகளை சரிபார்க்கவும், "
              "அருகிலுள்ள ஏரிகளை தூர்வாரவும். ஒன்றிய வளர்ச்சி அலுவலருக்குத் தெரிவிக்கவும்.",
    },
    "Medium": {
        "en": "Monitor weekly. Promote rainwater harvesting and water saving.",
        "ta": "வாரந்தோறும் கண்காணிக்கவும். மழைநீர் சேகரிப்பு மற்றும் நீர் சேமிப்பை ஊக்குவிக்கவும்.",
    },
    "Low": {
        "en": "No action needed. Continue routine monitoring.",
        "ta": "நடவடிக்கை தேவையில்லை. வழக்கமான கண்காணிப்பைத் தொடரவும்.",
    },
}


# ---------- Live data ----------
def fetch_series(lat, lon, start, end):
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": round(lon, 3),
        "latitude": round(lat, 3),
        "start": start,
        "end": end,
        "format": "JSON",
    }
    for _ in range(3):
        try:
            resp = requests.get(POWER_URL, params=params, timeout=90)
            resp.raise_for_status()
            data = resp.json()["properties"]["parameter"]["PRECTOTCORR"]
            s = pd.Series(data)
            s.index = pd.to_datetime(s.index, format="%Y%m%d")
            return s.where(s >= 0)
        except Exception:
            time.sleep(2)
    raise RuntimeError("NASA POWER request failed")


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def load_live_rain(blocks):
    req_end = pd.Timestamp.today().normalize() - pd.Timedelta(days=LATENCY_DAYS)
    start = f"{req_end.year - 10}0101"
    end = req_end.strftime("%Y%m%d")

    def job(row):
        return row.block_en, fetch_series(row.lat, row.lon, start, end)

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(job, blocks.itertuples()))
    fetched_at = pd.Timestamp.now().strftime("%d %b %Y, %H:%M")
    return dict(results), fetched_at


def build_rain_table(blocks, series):
    end = min(s.last_valid_index() for s in series.values())
    start = end - pd.Timedelta(days=89)
    rows = []
    for _, r in blocks.iterrows():
        s = series[r["block_en"]]
        recent = s[start:end].sum()
        normals = []
        for k in range(1, 11):
            a = start - pd.DateOffset(years=k)
            b = end - pd.DateOffset(years=k)
            normals.append(s[a:b].sum())
        normal = sum(normals) / len(normals)
        deficit = max(0.0, (normal - recent) / normal * 100) if normal > 0 else 0.0
        rows.append({
            "block_en": r["block_en"],
            "rain_90d_mm": round(recent, 1),
            "normal_90d_mm": round(normal, 1),
            "rainfall_deficit_pct": round(deficit, 1),
        })
    return pd.DataFrame(rows), end


# ---------- Risk model ----------
def norm(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0


@st.cache_data(show_spinner=False)
def compute_risk(df):
    df = df.copy()
    df["gw_stage_pct"] = df["gw_stage_pct"].fillna(0)
    df["gw_risk"] = (df["gw_stage_pct"] / 100).clip(0, 1)
    df.loc[df["gw_saline"] >= 1, "gw_risk"] = 1.0
    partial = (df["gw_saline"] > 0) & (df["gw_saline"] < 1)
    df.loc[partial, "gw_risk"] = df.loc[partial, "gw_risk"].clip(lower=0.5)

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
    return df


# ---------- Text helpers ----------
def reason_text(row, L):
    parts = []
    if L == "en":
        if row["rainfall_deficit"] > 10:
            parts.append(f"Rainfall {row['rainfall_deficit']:.0f}% below normal")
        if row["rain_90d_mm"] < 180:
            parts.append(f"Only {row['rain_90d_mm']:.0f} mm in last 90 days")
        if row["gw_saline"] > 0:
            parts.append("Groundwater saline (quality problem)")
        return ". ".join(parts) if parts else "Rainfall normal, groundwater stable"
    if row["rainfall_deficit"] > 10:
        parts.append(f"மழை இயல்பை விட {row['rainfall_deficit']:.0f}% குறைவு")
    if row["rain_90d_mm"] < 180:
        parts.append(f"கடந்த 90 நாட்களில் {row['rain_90d_mm']:.0f} மி.மீ மட்டுமே")
    if row["gw_saline"] > 0:
        parts.append("நிலத்தடி நீர் உவர்ப்பு (தரப் பிரச்சனை)")
    return ". ".join(parts) if parts else "மழை இயல்பு, நிலத்தடி நீர் நிலையாக உள்ளது"


def gw_text(row, t):
    if row["gw_saline"] >= 1:
        return t["gw_saline"]
    if row["gw_saline"] > 0:
        return f"{t['gw_partial']} ({row['gw_stage_pct']:.1f}%)"
    return f"{t['gw_safe']} {row['gw_stage_pct']:.1f}%"


# ---------- UI ----------
lang_choice = st.sidebar.radio("Language / மொழி", ["English", "தமிழ்"])
L = "en" if lang_choice == "English" else "ta"
t = TXT[L]
name_col = "block_en" if L == "en" else "block_ta"

if st.sidebar.button(t["refresh"]):
    st.cache_data.clear()
    st.rerun()

st.title(t["title"])
st.info(t["note"])

blocks = pd.read_csv("data/blocks.csv", encoding="utf-8")

try:
    with st.spinner(t["loading"]):
        series, fetched_at = load_live_rain(blocks)
    rain_df, end_date = build_rain_table(blocks, series)
    try:
        rain_df.to_csv("data/rainfall.csv", index=False, encoding="utf-8")
    except Exception:
        pass
    st.success(f"{t['live_ok']} {end_date.strftime('%d %b %Y')} "
               f"({t['fetched']} {fetched_at})")
except Exception:
    st.warning(t["live_fail"])
    try:
        rain_df = pd.read_csv("data/rainfall.csv", encoding="utf-8")
    except Exception:
        st.error("No rainfall data available. Run: python model\\fetch_rainfall.py")
        st.stop()

df = blocks.merge(
    rain_df[["block_en", "rain_90d_mm", "normal_90d_mm", "rainfall_deficit_pct"]],
    on="block_en",
)
df = compute_risk(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric(t["total"], len(df))
c2.metric(t["high"], int((df["risk_level"] == "High").sum()))
c3.metric(t["medium"], int((df["risk_level"] == "Medium").sum()))
c4.metric(t["low"], int((df["risk_level"] == "Low").sum()))

left, right = st.columns([2, 1])

with left:
    m = folium.Map(location=[9.38, 78.75], zoom_start=9)
    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=14,
            color=COLORS[r["risk_level"]],
            fill=True,
            fill_color=COLORS[r["risk_level"]],
            fill_opacity=0.8,
            tooltip=f'{r[name_col]} - {t["levels"][r["risk_level"]]}',
        ).add_to(m)
    st_folium(m, width=700, height=500)

with right:
    st.subheader(t["details"])
    chosen = st.selectbox(t["select"], df[name_col])
    v = df[df[name_col] == chosen].iloc[0]
    lvl = v["risk_level"]
    st.write(f"**{t['villages']}:** {v['panchayat_villages']}")
    st.write(f"**{t['risk_level']}:** {t['levels'][lvl]}")
    st.write(f"**{t['risk_score']}:** {v['risk_score']}")
    st.write(f"**{t['deficit']}:** {v['rainfall_deficit']}%")
    st.write(f"**{t['rain']}:** {v['rain_90d_mm']} mm")
    st.write(f"**{t['gw']}:** {gw_text(v, t)}")
    st.write(f"**{t['conf']}:** {int(v['confidence'] * 100)}%")
    st.write(f"**{t['reason']}:** {reason_text(v, L)}")

    st.markdown(f"**{t['reco']}**")
    box = {"High": st.error, "Medium": st.warning, "Low": st.success}[lvl]
    box(RECO[lvl][L])

    st.markdown(f"**{t['alert']}**")
    if L == "en":
        msg = (f"ALERT: {v[name_col]} block has {lvl.upper()} water stress "
               f"risk. {RECO[lvl]['en']}")
    else:
        msg = (f"எச்சரிக்கை: {v[name_col]} ஒன்றியம் - நீர் பற்றாக்குறை ஆபத்து: "
               f"{t['levels'][lvl]}. {RECO[lvl]['ta']}")
    st.code(msg, language=None)

st.subheader(t["table"])
high = df[df["risk_level"] == "High"].sort_values("risk_score", ascending=False)
show = high[[name_col, "risk_score", "rainfall_deficit"]].copy()
show.columns = t["cols"]
st.dataframe(show, width="stretch")