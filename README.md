# Dynasty Trade Finder

A free dynasty fantasy football trade tool that syncs with your **Sleeper** leagues and uses live player values from **FantasyCalc** to generate balanced trade proposals — including draft picks.

🔗 **[Use it live →](https://YOURUSERNAME.github.io/dynasty-trade-finder/)** *(update this URL after deploying)*

---

## Features

- **Sleeper Sync** — Enter your username to pull all your leagues, rosters, and traded picks directly from the Sleeper API
- **Live Dynasty Values** — Player values fetched from FantasyCalc's free API (no key needed), derived from 2M+ real trades
- **Draft Pick Values** — 2025 confirmed slots, 2026 per-slot values (1.01 ≠ 1.10), 2027+ Early/Mid/Late estimates
- **Pick Editor** — Manually correct future pick positions since Sleeper's API doesn't expose unfinished draft order
- **Trade Generator** — Finds balanced trade proposals within 12% value tolerance, searching all combinations of players and picks
- **Rebuild Analyzer** — Competitive outlook, strategy recommendations (Compete / Sell High / Rebuild), sell candidates, buy targets, and suggested trades
- **Superflex + 1QB** — Toggle between formats; values adjust accordingly
- **No Backend** — Runs entirely in your browser. No accounts, no API keys, no data stored anywhere

## How to Use

1. Open the site
2. Enter your Sleeper username and pick the season/format
3. Select a league — values load automatically
4. Click your team, then click a trade partner's team
5. Select up to 3 players/picks you want from them
6. Click **Find Balanced Trades** to see proposals

## Data Sources

| Source | What it provides | API Key? |
|--------|-----------------|----------|
| [Sleeper API](https://docs.sleeper.com) | Leagues, rosters, users, draft order, traded picks | No |
| [FantasyCalc API](https://fantasycalc.com) | Dynasty player values + draft pick values from real trades | No |

## Draft Pick Notes

- **2025**: Slot assignments come from the Sleeper draft's `slot_to_roster_id` field (confirmed)
- **2026**: Per-slot values from FantasyCalc (e.g., "2026 Pick 1.01" = 7065). Slot order is projected by roster strength — use the **📝 Edit Picks** tool to correct
- **2027+**: Estimated as Early / Mid / Late tiers based on roster value since exact positions are unknown
- Pick ownership from trades is applied via Sleeper's `traded_picks` endpoint

## Hosting / Deployment

This is a single `index.html` file. To deploy:

1. Fork or clone this repo
2. Go to **Settings → Pages → Source** → select `main` branch
3. Your site is live at `https://yourusername.github.io/dynasty-trade-finder/`

Or drag `index.html` onto [Netlify Drop](https://app.netlify.com/drop) for instant hosting.

## Updating

To update player values: just refresh the page — FantasyCalc values are fetched live every time you load a league.

To update the tool itself: edit `index.html` and push to GitHub. Changes go live in ~1 minute via GitHub Pages.

## Built With

- Vanilla HTML/CSS/JavaScript (no frameworks, no build step)
- [Sleeper API](https://docs.sleeper.com)
- [FantasyCalc API](https://fantasycalc.com)
- [Google Fonts (DM Sans + DM Mono)](https://fonts.google.com)

## License

MIT — free to use, modify, and share.
