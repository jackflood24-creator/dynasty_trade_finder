# Dynasty Trade Finder

Free dynasty fantasy football trade tool — syncs with **Sleeper**, uses live **FantasyCalc** values, generates balanced trade proposals, tracks value momentum, and runs ML predictions via GitHub Actions.

🔗 **[Use it live →](https://YOURUSERNAME.github.io/dynasty-trade-finder/)** *(update URL after deploying)*

## Features

### 🔄 Trade Finder
- Sleeper sync — enter your username to pull leagues, rosters, traded picks
- Live FantasyCalc values (no API key), including per-slot draft pick values
- Pick editor to correct future pick ownership
- Balanced trade proposals within 12% value tolerance

### 📈 Value Momentum
- Your roster's 30-day value trend from FantasyCalc
- League-wide risers & fallers
- Buy Low / Sell High / Hold signals (heuristic + ML when available)
- Position filters (QB / RB / WR / TE)

### 🔄 Rebuild Analyzer
- Power ranking visualization
- Strategy recommendation (Compete / Sell High / Rebuild)
- Sell candidates + buy targets with trade suggestions

### 🤖 ML Predictions (via GitHub Actions)
- XGBoost models trained per position on nflfastr usage trends
- Runs weekly on a schedule — no local Python needed
- Predictions auto-committed to `data/predictions.json`
- Web tool loads predictions and shows 🤖 model signals

## Setup

### 1. Deploy the site

1. Create a new GitHub repo called `dynasty-trade-finder`
2. Upload ALL files from this project (keep the folder structure!)
3. Go to **Settings → Pages → Source** → `main` branch, `/ (root)` → Save
4. Site is live at `https://yourusername.github.io/dynasty-trade-finder/`

### 2. Enable ML predictions (optional)

The GitHub Action runs automatically every Tuesday at 3am EST. To trigger it manually:

1. Go to your repo → **Actions** tab
2. Click **"Update Dynasty Predictions"** workflow
3. Click **"Run workflow"** → **"Run workflow"**
4. Wait ~2 minutes — predictions appear in `data/predictions.json`
5. The site auto-loads them on the Momentum page

**Important:** Go to **Settings → Actions → General** and make sure "Read and write permissions" is enabled under "Workflow permissions". This lets the Action commit the predictions file.

## File Structure

```
dynasty-trade-finder/
├── index.html                          # The web app (all you need for basic use)
├── data/
│   └── predictions.json                # ML predictions (auto-updated by GitHub Action)
├── scripts/
│   └── generate_predictions.py         # Prediction pipeline (runs in GitHub Actions)
├── .github/
│   └── workflows/
│       └── update-predictions.yml      # GitHub Action config
├── dynasty_exploration.py              # Optional: local data exploration
├── dynasty_model_pipeline.py           # Optional: local model training
├── requirements.txt                    # Python dependencies
├── README.md
└── LICENSE
```

## Data Sources

| Source | Provides | Key? |
|--------|----------|------|
| [Sleeper API](https://docs.sleeper.com) | Leagues, rosters, draft order, traded picks | No |
| [FantasyCalc API](https://fantasycalc.com) | Dynasty values, pick values, 30-day trends | No |
| [nflfastr](https://github.com/nflverse/nflverse-data) | Weekly player stats (targets, carries, etc.) | No |

## License

MIT — free to use, modify, and share.
