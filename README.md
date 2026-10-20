# AI Water Scarcity Prediction System: Ramanathapuram District

A bilingual (Tamil / English) dashboard that ranks all 11 blocks of Ramanathapuram
district by water-stress risk. It downloads the latest weather data when opened,
recalculates the risk, and shows the reason, a recommended action and an
SMS-style alert message for each block.

> **Hackathon problem statement 8:** AI Water Scarcity Prediction System.
> The prototype works at **block level (11 blocks, 429 panchayat villages)**.
> The same pipeline can move to village level once village-level data is available.

**Live demo:** https://ramnad-water-scarcity.streamlit.app

## What it does

- Colour-coded map of all 11 blocks (Low / Medium / High risk)
- Rainfall, temperature and humidity downloaded live from NASA POWER
  (cached for 6 hours, with a "Refresh data now" button and a "data up to" date)
- Block details: rainfall deficit, 90-day rain, temperature vs normal, groundwater
  status, tank water spread, estimated drinking-water need, model confidence
- Plain-language reason for each risk level, in Tamil and English
- Recommended action and an alert message preview (with estimated tanker loads)
- If the live download fails, the last saved weather data is shown with a warning

## Data sources

| Data                                                     | Source                                                                                            | Real or estimated                             | Update                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------- |
| Rainfall, temperature, humidity (daily, last 10 years)   | NASA POWER, about 50 km grid                                                                      | Real (satellite / model based)                | **Live download**, a few days delay                 |
| Groundwater stage of extraction and Safe / Saline status | CGWB, _Dynamic Groundwater Resources of Tamil Nadu 2024_ (firka level)                            | Real, matched to blocks by name (approximate) | Yearly report                                       |
| Tank water spread                                        | Sentinel-2 imagery and JRC Global Surface Water, processed in Google Earth Engine                 | Satellite estimate (not depth)                | Snapshot (45 days to 20 Oct 2026), re-run to update |
| Block list and panchayat village counts                  | TNAU Agritech district block list                                                                 | Real                                          | Static                                              |
| Rural population (Census 2011)                           | TN Rural Development Dept, panchayat union table                                                  | Real (Mandapam derived, see below)            | Static                                              |
| Drinking-water need                                      | Population x 55 litres per person per day (Jal Jeevan Mission norm); tanker size 12,000 L assumed | **Estimate**                                  | Static                                              |
| Block coordinates                                        | Approximate block centres                                                                         | Estimated                                     | Static                                              |
| Tamil block names                                        | Entered manually                                                                                  | To be verified against the official list      | Static                                              |

**CGWB finding for Ramanathapuram (2024):** 38 firkas assessed. 29 are Safe and
9 are Saline (Kadaladi, Sayalgudi, S.Tharaikudi, Thirupullani, Thondi,
Mangalagudi, Sikkal, Melaselvanur, Mudhukulathur South). No firka is
Semi-critical, Critical or Over-exploited. In this district the groundwater
problem is mainly **quality (salinity)**, not over-extraction.

**Population note:** The Mandapam value (about 75,828) is derived from the district
rural total minus the other ten panchayat unions, because its row could not be
read from the source PDF.

## Method

1. **Rainfall:** last 90 days compared with the average of the same window in the
   previous 10 years. Output: rainfall deficit %.
2. **Temperature:** last 90 days mean compared with the same 10-year window
   (anomaly in degrees C).
3. **Groundwater:** stage of extraction from the CGWB table. Saline blocks are
   treated as high groundwater risk (water not usable).
4. **Tank:** fraction of seasonal water pixels that are wet in the last 45 days,
   compared with the average of the same window in the previous 3 years.
   Used only for inland blocks with enough clear images.
5. **Risk score (0 to 1):** weighted sum: 0.30 rainfall deficit, 0.15 low recent rain,
   0.25 groundwater, 0.10 heat, 0.20 tank. The weights are our assumption, not a
   published formula. If an input is missing for a block, the remaining weights
   are rescaled.
6. **Risk level:** blocks are ranked and split into Low / Medium / High thirds.
   This is a **relative ranking among the 11 blocks**, not an absolute drought level.
7. **Model:** a Random Forest trained on these proxy labels gives the level and
   a confidence value.
8. **Recommendations and alerts:** rule-based text per risk level.

## Limitations (please read)

- **Block level, not village level.** Village-level tank, consumption and
  groundwater data is not public.
- **Only weather is live.** Groundwater is a yearly report. The tank data is a
  snapshot, and population is from Census 2011.
- **Tank values are indicative.** Satellite water spread is not depth. Blocks
  with a very large water area (Kadaladi, Mandapam, Tiruppullani,
  Ramanathapuram, Rajasingamangalam) are excluded from the risk score because
  tidal water and salt pans dominate. Cloud cover in the monsoon season reduces
  reliability.
- **No ground-truth labels.** The model learns our proxy score, so we do not
  report an accuracy number. Confidence is the model's own probability.
- **Weather resolution is about 50 km**, so neighbouring blocks can share values.
- **Groundwater is firka data matched to blocks by name.** Firka and block
  boundaries differ, so this is an approximation.
- **Water need is an estimate** (rural population only, 2011).
- Block coordinates and Tamil names need checking against official sources.
- Alerts are previews only. No SMS is sent.

## Future work

- Village-level version (429 panchayat villages) with finer rainfall (CHIRPS, about 5 km)
- Tank water spread per tank (village tank polygons) and automatic refresh
- Validate predictions against panchayat reports, tanker records and well observations
- Real SMS / WhatsApp alerts to block officers

## Project structure

```
water-scarcity/
├── app/main.py                  Dashboard (live download + risk model)
├── data/blocks.csv              Blocks, Tamil names, groundwater, population
├── data/rainfall.csv            Last saved weather (fallback)
├── data/tank_water.csv          Satellite tank water spread (snapshot)
├── model/                       Offline helper scripts
└── requirements.txt
```

## How to run (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app\main.py
```

An internet connection is needed for the live weather download.

## Team

**Team name:** Zenith
