# Dynasty Trade Finder — Project Handoff

Paste this entire document at the start of a new Claude chat (or share with
Claude Code when you open the repo) to bring the assistant up to speed.

---

## Project overview

A dynasty fantasy football trade finder web tool, deployed on **Firebase Hosting**.

- **Single-file architecture:** `index.html` (~3,000 lines) containing HTML,
  CSS, and JavaScript. Uses vanilla JS with a global `S` state object — no
  framework, no build step.
- **APIs:** Sleeper (rosters, users, picks, drafts) and FantasyCalc (player
  values).
- **Backend:** Python ML pipeline at `scripts/generate_predictions.py`,
  triggered **daily** by GitHub Actions. Writes `predictions.json` to the repo.
- **My league:** superflex, 10-team. Sleeper username: `jackflood`. League
  name: `Arbitrage Fantasy League (AFL)`.

## Deployment

- **Firebase project ID:** `dynasty-trade-analyzer`
- **Live URL:** `https://dynasty-trade-analyzer.web.app`
- **GitHub repo:** `https://github.com/jackflood24-creator/dynasty_trade_finder`
- **Deploy workflow:** `.github/workflows/deploy.yml` — triggers on every push
  to `main`, uses `FIREBASE_TOKEN` GitHub secret
- **Predictions workflow:** `.github/workflows/update-predictions.yml` — runs
  daily at 8am UTC, commits updated `data/predictions.json`, which in turn
  triggers a redeploy

### Local workflow (all edits go here)

```
~/Desktop/dynasty_trade_finder_git/   ← cloned repo, push from here
~/Desktop/dynasty-trade-analyzer/     ← old local folder, ignore
```

## Features in the app

- Rosters page with bidirectional trade finder
- Momentum tracker (FantasyCalc trend30Day)
- Rebuild analyzer with tier-aware suggestions
- Picks editor (per-team slot overrides)
- Trade history with pick resolution
- Signal badges (BUY / SELL / HOLD+ / AVOID)
- **League switcher** in header — switch leagues without re-entering username

## Recent work (already implemented in `index.html`)

### Firebase migration (2026-05-17)

Migrated from GitHub Pages to Firebase Hosting (Option 1 — hosting only).

- `firebase.json` — serves from repo root, `index.html` as SPA fallback
- `.firebaserc` — pins project `dynasty-trade-analyzer`
- `.github/workflows/deploy.yml` — auto-deploys on push to `main`
- Zero code changes to `index.html` for the migration itself

### League switcher (2026-05-17)

Added `⇄ <League Name>` button in the header (next to Reset).

- Visible on all pages except the connect screen and league picker
- Clicking it clears league-specific state (rosters, picks, vals) but keeps
  username and leagues list — no re-login needed
- Calls `renderLeagues()` + `go('lea')` to show the picker immediately
- Button label updates to the current league name after selection
- New function: `switchLeague()` (near `resetApp()`)
- `go()` updated to show/hide `#bswl` button

### Stale roster fix (2026-05-17)

Root cause was two-part:

1. **Wrong season** — app defaulted to `S.season = '2025'`. Dynasty activity
   (trades, rookie draft) happens in the current Sleeper season (2026). Fixed
   by defaulting to `'2026'` and adding a 2026 button to the season toggle.
2. **Sleeper server-side caching** — `cache: 'no-store'` on fetch wasn't
   enough; Sleeper's servers were returning stale responses. Fixed by appending
   `?_t=Date.now()` to all Sleeper API URLs except `/players/nfl` (which is
   large and only loaded once per session via `playersLoaded` guard).

### Completed draft picks fix (2026-05-17)

`buildPickAssets` was creating pick entries for all years with FC data,
including years where the draft is already `complete`. This caused 2026 picks
to appear on rosters even though those picks had been used to draft rookies.

Fix: skip building pick assets for any year where `draftStatusByYear[year] === 'complete'`.
Applied in both slot-based (5A) and bucket-based (5B) build loops.

Also suppressed `⚠ MISSED` log warnings for traded picks in completed draft
years — these misses are expected and not errors.

### Trade history: top 10 best trades (2026-05-17)

Replaced the two-column "Most Value Gained / Most Value Lost" layout with a
single full-width "Top 10 Best Trades" section ranked by value gained.
"Most Value Lost" was removed as it wasn't meaningful.

### Signals description fix (2026-05-17)

The hardcoded message "signals use XGBoost model trained on usage trends" was
wrong — the pipeline was running in fallback mode ("FantasyCalc trends + age
curves, no nflfastr model"). Fixed to read `S.predictions.model_info.note`
directly from the JSON and show "refreshes daily" instead of a static label.

### Daily predictions pipeline (2026-05-17)

Changed `update-predictions.yml` cron from `0 8 * * 2` (weekly Tuesday) to
`0 8 * * *` (daily 8am UTC). Each daily run commits new `data/predictions.json`
which automatically triggers a Firebase redeploy via `deploy.yml`.

---

## Trade Finder UX/UI overhaul (prior work)

1. Nav tab renamed: `🤝 Rosters` → `🤝 Rosters + Trade Finder`
2. Two-panel layout when a target is selected:
   - LEFT (red border): "You'll send" — clicking your players locks them in
   - RIGHT (blue border): "You want" — target's roster
3. Live balance bar showing send vs receive totals + overpay/underpay %
4. Three trade styles: Balanced (±12%), Flexible (±20%), Big Swap (≥40% pieces)
5. Max-per-side toggle: 3 or 4 assets
6. 🎲 Shuffle button — banded randomization for variety
7. State additions: `sendPids`, `tradeStyle`, `tradeMax`, `tradeShuffle`

### Bidirectional `findTrades` algorithm

The core trade-finding function was rewritten three times to get right.
Final design (currently in the file):

- Builds asset pools from BOTH sender's and target's rosters
- Enumerates every sweetener subset on each side (size 0 to maxSlots)
- Pairs every send-subset with every receive-subset — true bidirectional search
- Scores each pairing by closeness + complexity tiebreaker (simpler trades win
  on ties)
- Tolerance is a labeling concept only, NOT a filter — algorithm always
  returns top 25 closest combos regardless of style
- Algorithm-suggested items tagged `_suggested:true`; UI renders them with
  a 💡 icon, italics, faint amber bg
- Locked items have no special marker (just the plain pRow render)

---

## Things discussed but NOT yet implemented

### Firebase options not taken

2. **Hosting + Cloud Functions** as an API proxy for Sleeper/FantasyCalc.
   Requires Blaze plan but generous free tier. Worth it if rate-limited.
3. **Hosting + Functions + Firestore.** Move predictions and any sample data
   to Firestore. Unlocks per-user features (saved trades, watchlists). Needs
   Firebase Auth.
4. **Framework rewrite.** Not recommended — the single-file architecture is a
   feature.

---

## Things to watch out for

- The `S` global state object holds everything. Don't break the contract:
  `myRid`, `tgtRid`, `tgtPids`, `sendPids`, `rosters`, `users`, `vals`,
  `picks`, `pickVals`, `valsLoaded`, `players`, `tradedPicks`, `drafts`,
  `fcMeta`, `slotOverrides`.
- Pick values come from `S.picks[id].value` (computed from FantasyCalc
  pick rank values + slot overrides in the Picks editor).
- Player values come from `gv(id)` — wraps `S.vals[id]` with a fallback.
- `getPicksForRoster(rid)` returns the picks owned by a roster after
  resolving traded picks.
- Render order matters: `loadPlayers` → `loadFantasyCalcValues` →
  `buildPickAssets` → `renderRosters`. Never call render before values
  load or you'll show "—" everywhere.
- Mobile breakpoint: 700px. The dual trade panel stacks vertically below.
- Browser support: vanilla ES5-ish JS (`var`, `function`, no arrow functions
  in production code). Don't add modern syntax without checking.
- **Season default is `2026`** — update each new year in both `S.season` init
  and the season toggle buttons in the connect screen HTML.
- **Completed drafts** — `buildPickAssets` skips years where
  `draftStatusByYear[year] === 'complete'`. When a new season's rookie draft
  finishes, picks disappear automatically and the drafted players show instead.
- **Sleeper cache-busting** — all Sleeper API calls except `/players/nfl`
  append `?_t=Date.now()` to prevent server-side stale responses.

## File layout

```
/
├── index.html                          # The app (~3000 lines)
├── firebase.json                       # Firebase Hosting config
├── .firebaserc                         # Firebase project binding
├── data/
│   └── predictions.json                # Output of ML pipeline (daily)
├── scripts/
│   └── generate_predictions.py         # Daily ML job
└── .github/
    └── workflows/
        ├── deploy.yml                  # Auto-deploy to Firebase on push
        └── update-predictions.yml      # Daily predictions + triggers deploy
```

## How to test locally

```bash
# Serve from the repo root — index.html needs http:// to fetch data/*.json
python -m http.server 8000
# Open http://localhost:8000
```

## How to deploy

Any `git push origin main` from `~/Desktop/dynasty_trade_finder_git/` triggers
the deploy workflow automatically. To manually trigger predictions:
GitHub → Actions → "Update Dynasty Predictions" → Run workflow.

## What I want to work on next

(Fill this in when you start the new chat.)
