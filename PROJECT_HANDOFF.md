# Dynasty Trade Finder — Project Handoff

Start every new session by telling Claude Code:
> "Read PROJECT_HANDOFF.md, then [your task]."

Run Claude Code from `~/Desktop/dynasty_trade_finder_git/` so it can read files and run git without extra path work.

---

## Project overview

A dynasty fantasy football trade finder web tool, deployed on **Firebase Hosting**.

- **Single-file architecture:** `index.html` (~4200 lines) containing HTML, CSS, and JavaScript. Uses vanilla JS with a global `S` state object — no framework, no build step. No arrow functions in production code.
- **APIs:** Sleeper (rosters, users, picks, drafts, transactions) and FantasyCalc (player values, ranks, trends).
- **Backend:** Python ML pipeline at `scripts/generate_predictions.py`, triggered **daily** by GitHub Actions. Writes `data/predictions.json` to the repo.
- **My league:** superflex, 10-team. Sleeper username: `jackflood`. League name: `Arbitrage Fantasy League (AFL)`.

## Deployment

- **Firebase project ID:** `dynasty-trade-analyzer`
- **Live URL:** `https://dynasty-trade-analyzer.web.app`
- **GitHub repo:** `https://github.com/jackflood24-creator/dynasty_trade_finder`
- **Deploy workflow:** `.github/workflows/deploy.yml` — triggers on every push to `main`, uses `FIREBASE_TOKEN` GitHub secret
- **Predictions workflow:** `.github/workflows/update-predictions.yml` — runs daily at 8am UTC, commits updated `data/predictions.json`, which triggers a redeploy

### Local workflow

```
~/Desktop/dynasty_trade_finder_git/   ← cloned repo, push from here
~/Desktop/dynasty-trade-analyzer/     ← old local folder, ignore
```

```bash
python -m http.server 8000   # serve locally, open http://localhost:8000
```

Any `git push origin main` auto-deploys to Firebase.

---

## Features in the app

**Primary nav (row 1):**
- Rosters + Trade Finder — bidirectional trade proposals
- Momentum — FantasyCalc trend30Day tracker
- Rebuild Analyzer — tier-aware trade suggestions
- Trade History — top 10 best trades + timeline
- Edit Picks — per-team slot overrides
- H2H Record — cross-season head-to-head stats

**Secondary nav (row 2):**
- Waiver Wire Targets — adds/drops activity feed across your league
- Trade Offer Evaluator — add players or picks to two panels, get instant value verdict
- Draft Grade — rookie draft grades, relative to your league's draft class
- Watch List — save players to track with localStorage persistence
- All Leagues Dashboard — multi-league summary (dynasty/keeper only)
- League Activity Feed — recent trades/adds/drops in this league
- Age Curve Projector — 5-year value projection with compound decay per player on your roster
- Dynasty Rankings — FantasyCalc dynasty rank order, position filter, FA/owner tags

**Header:**
- League Switcher `⇄` button — clears league state, returns to league picker without re-login

---

## Architecture constants to never break

- **`S` global state contract:** `myRid`, `tgtRid`, `tgtPids`, `sendPids`, `rosters`, `users`, `vals`, `picks`, `pickVals`, `valsLoaded`, `players`, `tradedPicks`, `drafts`, `fcMeta`, `slotOverrides`, `watchList`
- **`gv(id)`** — gets player value with fallback; never read `S.vals[id]` directly
- **`getPicksForRoster(rid)`** — returns picks owned by a roster after resolving traded picks
- **Pick values:** `S.picks[id].value` (slot-based from FantasyCalc + slot overrides)
- **`S.pickVals` key format:** `"2026 Pick 1.01"` (not bare `"1.01"`) — matches FantasyCalc API
- **`S.fcMeta[sleeperId]`:** `{trend, age, yoe, rank, posRank, tier, stdDev, team, pos}` — `rank`/`posRank` are FC dynasty ranks
- **Render order:** `loadPlayers` → `loadFantasyCalcValues` → `buildPickAssets` → `renderRosters`; never render before values load
- **Season default:** `S.season = '2026'` — update each year in both `S.season` init and season toggle buttons
- **Sleeper cache-busting:** all Sleeper API calls except `/players/nfl` append `?_t=Date.now()`
- **Completed drafts:** `buildPickAssets` skips years where `draftStatusByYear[year] === 'complete'`
- **Dynasty filter:** `isDynastyLeague(l)` checks `l.settings.type === 2` (dynasty) or `1` (keeper); type 0/missing = redraft. Redraft leagues are hidden from the league picker and dashboard.
- **Mobile breakpoint:** 700px — dual trade panel stacks vertically
- **`REL_POS`:** `new Set(['QB','RB','WR','TE'])` — relevant positions for most features

---

## Recent work

### Bidirectional trade finder (2026-05-17)

The `findTrades` function was rewritten to be truly bidirectional. Previously the receive side was always fixed to exactly what the user clicked. Now both sides can have algorithm-suggested sweeteners:

- `anchorSend` (your locked players) + `sendFiller` from your pool
- `anchorReceive` (what you clicked) + `recvFiller` from **their** pool

**Algorithm:** Pre-enumerates all receive-filler subsets (~900 max), sorts by value, then for each send-filler combo binary-searches for nearest receive match — no O(N²) cross-product. Up to 2,500 pairs explored per call.

**Signature change:** `findTrades(myRoster, targetRoster, targetIds, opts)` — `targetRoster` is now the second argument.

**Randomizer:** Both pools shuffled independently with smaller bands (3-4 vs 6), larger sort jitter (6% vs 3%) — repeated shuffles surface different trade structures.

**Rendering:** Suggested items on either side show with 💡 icon and italic. `pRow(a)` reads `a._suggested`.

### Draft grade fixes (2026-05-17)

Three bugs fixed in `renderDraftGrade`:

1. **Attribution:** Was using `slot_to_roster_id[pk.draft_slot]` (original slot owner before pick trades). Fixed to use only `pk.picked_by` → map to roster via `S.rosters[j].owner_id`.

2. **Pick cost lookup:** `S.pickVals` stores FC data as `"2026 Pick 1.01"` but code was looking up `"1.01"` (always 0). Fixed to use `draft.season + ' Pick ' + round + '.' + slotPadded`. For completed drafts where FC no longer carries the slots, falls back to exponential ADP curve: `7500 * e^(-0.08 * (pickNo-1))`.

3. **Grading:** Absolute thresholds (-4% to +8% = B) caused everyone to cluster. Replaced with z-score relative grading vs this draft class: z≥1.5=A+, z≥0.8=A, z≥0.25=B+, z≥-0.25=B, z≥-0.8=C, z≥-1.5=D, else F. Per-pick STEAL/REACH badges when surplus/deficit exceeds 20% of avg pick cost.

### Age curve fixes (2026-05-17)

Two bugs fixed:

1. **Compound decay:** `projectValue` was applying decay to original `val` each year (`val * (1-dr)`) instead of compounding from accumulated `v`. Fixed to `v * (1-dr)`. RBs now show real cliff-shaped decline.

2. **Per-player bar scale:** Bars were scaled to global roster max. Fixed to scale each player to their own current value = 100%. Added a "Now" column as baseline so the trajectory is visible.

### Trade history redundancy removed (2026-05-17)

Both sides of each trade showed "+/- since trade" — always equal and opposite. Removed from both `tradeCard` and `timelineRow`. Replaced with a single net verdict: "Team X ahead by Y at current values."

### Rankings aligned to FantasyCalc dynasty rank (2026-05-17)

Rankings page now sorts by `S.fcMeta[id].rank` (FantasyCalc overall dynasty rank, ascending) instead of raw value. Shows `QB1`/`RB4` position rank tags when "All" filter active. Matches FantasyPros dynasty ECR ~80-90% since FC uses similar methodology.

### 8 new secondary pages added (2026-05-17)

All in `index.html`, accessible via second nav row (`navtab2` class). Functions: `openWvrTargets`, `openEval`, `openDraftGrade`, `openWatchList`, `openDash`, `openActivity`, `openAgeCurve`, `openRankings`.

### Trade offer evaluator: pick support (2026-05-17)

`evalSearch` searches both `S.vals` (players) and `S.picks` (picks). `evalTogglePicks`, `evalShowPicks`, `evalAddPickKey` let users browse and add picks by round. `evalAddIdx` handles both player and pick types.

### Firebase migration + other fixes (prior session)

- Firebase Hosting, auto-deploy on push to `main`
- League switcher `⇄` header button (`switchLeague()` function)
- Season default changed to `2026`; `?_t=Date.now()` cache-busting on all Sleeper calls
- Completed draft picks excluded from `buildPickAssets` (`draftStatusByYear[year] === 'complete'`)
- Trade history changed to "Top 10 Best Trades" (removed "Most Value Lost")
- Signals description reads from `S.predictions.model_info.note` (was hardcoded)
- Daily predictions pipeline (cron `0 8 * * *`)

---

## Things NOT yet implemented / ideas on the table

- **Redraft league support** — user expressed interest; `isDynastyLeague()` already gates this
- **Firebase Cloud Functions** — API proxy for Sleeper/FantasyCalc if rate-limited (Blaze plan needed)
- **Firestore + Auth** — would unlock saved trades, watchlists synced across devices
- **Framework rewrite** — not recommended; single-file is a feature

---

## File layout

```
/
├── index.html                          # The entire app (~4200 lines)
├── firebase.json                       # Firebase Hosting config
├── .firebaserc                         # Firebase project binding
├── PROJECT_HANDOFF.md                  # This file
├── data/
│   └── predictions.json                # Daily ML output
├── scripts/
│   └── generate_predictions.py         # Daily ML job
└── .github/
    └── workflows/
        ├── deploy.yml                  # Auto-deploy on push to main
        └── update-predictions.yml      # Daily predictions → triggers deploy
```

---

## How to start a new session efficiently

1. **Open Claude Code from the repo root:**
   ```bash
   cd ~/Desktop/dynasty_trade_finder_git && claude
   ```

2. **First message — one shot:**
   > "Read PROJECT_HANDOFF.md, then [exact description of what you want]."

3. **Be specific about the bug or feature.** Vague = more back-and-forth = more tokens.
   - Bad: "the trade finder seems off"
   - Good: "the trade finder doesn't add sweeteners to the other team's side — only mine"

4. **State constraints upfront if unusual** (they're already in this doc, so normally you don't need to repeat them).

5. **Keep sessions to one feature cluster.** Mixing 5 unrelated changes fills context faster.

6. **End each session by updating this doc:** "Update PROJECT_HANDOFF.md with everything we did today."

## What I want to work on next

(Fill this in when you start the new chat.)
