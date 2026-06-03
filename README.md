# 🏆 FIFA World Cup 2026 — AI Match Prediction Engine

> *An end-to-end machine learning pipeline that predicts every group stage match, simulates the full knockout bracket, and crowns a World Cup champion — built on 150+ years of international football data.*

<br>

## 📊 Results at a Glance

| Metric | Score |
|--------|-------|
| Model Accuracy (held-out 2022+ test set) | **~56%** |
| Macro F1 Score | **~0.46** |
| Training matches used | **49,000+** |
| WC 2026 group fixtures predicted | **72** |
| Predicted Champion | **ESP SPAIN** |

> The model outperforms a naive "home team always wins" baseline and a random baseline (~33%) by a significant margin on a genuinely hard 3-class problem (Home Win / Draw / Away Win).

<br>

## 🗂️ Project Structure

```
wc2026-prediction/
│
├── wc2026_prediction.ipynb       ← Main notebook (run this)
│
├── data/
│   ├── WorldCupMatches.csv       ← Every WC match 1930–2022
│   ├── WorldCups.csv             ← Tournament-level summaries
│   └── results.csv               ← 49K+ international matches since 1872
│
├── outputs/
│   ├── wc2026_match_predictions.csv   ← All 72 group match predictions
│   └── wc2026_standings.csv           ← Predicted group standings
│
└── README.md
```

<br>

## 🧠 What This Project Does

### Pipeline Overview

```
Raw Data (49K+ matches)
        ↓
  ELO Rating System         ← all matches, tournament-weighted, with decay
        ↓
  Rolling Form Features     ← last 10 matches, competitive games only
        ↓
  Head-to-Head Records      ← historical win rates between any two teams
        ↓
  4 ML Classifiers trained  ← LR / RF / GBM / HistGBM
        ↓
  Best model selected       ← time-based eval on 2022+ data
        ↓
  72 Group Stage Predictions
        ↓
  Full Knockout Simulation  ← R32 → R16 → QF → SF → Final
        ↓
  🏆 Predicted Champion
```

<br>

## 📐 Feature Engineering

### 1. ELO Rating System (Custom)

Built from scratch on all 49,000+ international matches — not just World Cup games. Key design decisions:

| Feature | Value | Why |
|---------|-------|-----|
| Starting ELO | 1500 | Standard Elo baseline |
| K-factor (WC match) | 40 | WC results matter most |
| K-factor (Major tournament) | 30 | Euros, Copa America, AFCON etc. |
| K-factor (Qualifier) | 20 | Meaningful but not peak importance |
| K-factor (Friendly) | 10 | Low stakes, low signal |
| ELO decay factor | 0.98/yr | Inactive teams regress to 1500 |
| Home advantage | +50 pts | Applied only for non-neutral venues |
| Goal margin multiplier | Up to 2.5× | Bigger wins = bigger rating shift |

**Why this fixes the original model:** The v1 model only updated ELOs from World Cup matches. Turkey hadn't played a WC since 2002 — so their ELO was frozen, making them look deceptively strong. Now every qualifier, Nations League game, and friendly updates their rating correctly.

### 2. Rolling Form (Last 10 Matches)

For each team, at every match, we snapshot:
- **Win rate** — last 10 games
- **Goals for / against** — rolling average
- **Competitive win rate** — excludes friendlies

### 3. Head-to-Head Records

Historical win rate between every pair of teams, updated incrementally.

### Full Feature Set (16 features)

```python
FEATURES = [
    'home_elo_pre', 'away_elo_pre', 'elo_diff',      # ELO strength
    'home_form_wr', 'away_form_wr', 'form_wr_diff',  # Recent form
    'home_form_gf', 'home_form_ga',                  # Scoring/conceding
    'away_form_gf', 'away_form_ga',
    'home_form_comp', 'away_form_comp',              # Competitive form
    'h2h_home_wr',  'h2h_away_wr',                  # Head-to-head
    'is_neutral', 'is_wc',                           # Match context
]
```

<br>

## 🤖 Models Compared

| Model | Accuracy | Macro F1 | Log Loss |
|-------|----------|----------|----------|
| Logistic Regression | ~0.52 | ~0.43 | ~1.05 |
| Random Forest | ~0.55 | ~0.45 | ~1.02 |
| **Gradient Boosting** | **~0.56** | **~0.46** | **~0.99** |
| HistGradBoost | ~0.56 | ~0.46 | ~1.00 |

Train/test split is **time-based** (pre-2022 → train, 2022+ → test), which mimics real deployment and avoids data leakage.

<br>

## 📈 Visualizations

The notebook produces 10+ publication-quality visualizations:

- **Goals over time** — total and average per WC tournament
- **Match outcomes** — home win / draw / away win breakdown
- **Scoring efficiency scatter** — all teams ≥10 WC appearances
- **Top teams** — goals scored and WC appearances
- **Group vs knockout dynamics** — goals, margins, patterns
- **ELO leaderboard** — top 20 teams after 150 years of matches
- **Model comparison dashboard** — metrics, confusion matrices, ROC curves
- **Feature importance** — what the model actually uses
- **Group standings** — all 12 WC 2026 groups with expected points
- **Bracket path visualization** — top 8 teams through the knockout stages

<br>

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run

1. Clone the repo and put the 3 CSV files in the root directory
2. Open `wc2026_prediction.ipynb` in Jupyter / VS Code / Kaggle
3. Run all cells top to bottom

```bash
git clone https://github.com/yourusername/wc2026-prediction
cd wc2026-prediction
jupyter notebook wc2026_prediction.ipynb
```

### Data Sources

| File | Source |
|------|--------|
| `results.csv` | [Kaggle — International Football Results 1872–2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| `WorldCupMatches.csv` | [Kaggle — FIFA World Cup](https://www.kaggle.com/datasets/abecklas/fifa-world-cup) |
| `WorldCups.csv` | Same as above |

<br>

## 🔑 Key Fixes vs Naive Baseline

The original model predicted Turkey in the final 💀. Here's what was wrong and how we fixed it:

| # | Problem | Fix |
|---|---------|-----|
| 1 | ELO only updated from WC matches | ELO now uses all 49K+ international matches |
| 2 | Stale ELOs for inactive teams | Annual ELO decay — inactive teams regress to 1500 |
| 3 | All match types treated equally | Tournament-weighted K-factor (WC > Qualifier > Friendly) |
| 4 | Friendlies counted as competitive form | Competitive form tracked separately |
| 5 | Group standings: winner-takes-all | Probabilistic expected points |
| 6 | No historical matchup context | Head-to-head win rates as a feature |

<br>

## 🔭 What's Next

| Idea | Expected Impact |
|------|----------------|
| Monte Carlo simulation (N=10,000 runs) | Quantify uncertainty — "Brazil has 21% chance to win it all" |
| Poisson regression for score prediction | Predict scorelines, not just outcomes |
| FIFA World Rankings as a feature | Official current strength signal |
| Squad age / injury data | Captures human factors |
| XGBoost / LightGBM + hyperparameter tuning | ~1–2% F1 improvement |
| Streamlit web app deployment | Live dashboard with real WC 2026 scores |

<br>

## 📬 Connect

Built as part of a sports analytics portfolio. Open to data science / sports analytics roles and collaborations.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/aryan-bhargune-98519a346/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://kaggle.com/yourprofile)

---

*Data covers international matches from 1872 through mid-2026. WC 2026 group fixtures sourced from the results dataset.*
