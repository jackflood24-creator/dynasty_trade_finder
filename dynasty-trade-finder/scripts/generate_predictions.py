"""
Dynasty Prediction Generator (GitHub Actions)
==============================================
Runs on a schedule via GitHub Actions. Fetches current FantasyCalc values
and nflfastr stats, trains XGBoost models per position, and outputs
data/predictions.json for the web tool to consume.

Output format:
{
  "updated": "2025-03-25T08:00:00Z",
  "players": {
    "6794": {  // sleeper_id
      "name": "Justin Jefferson",
      "pos": "WR", "team": "MIN", "age": 26.8,
      "value": 8760, "trend": 372, "trendPct": 4.2,
      "predicted": 280, "signal": "HOLD+",
      "features": {"targets_avg": 9.2, "target_share_trend": 0.03, ...}
    }, ...
  },
  "model_info": {
    "QB": {"mae": 120, "top_features": ["passing_yards_avg", ...]},
    ...
  }
}
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

try:
    import nfl_data_py as nfl
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error
except ImportError as e:
    print(f"Missing: {e}")
    exit(1)

print("=" * 60)
print("DYNASTY PREDICTION GENERATOR")
print(f"Run time: {datetime.now(timezone.utc).isoformat()}")
print("=" * 60)

# ============================================================
# 1. LOAD FANTASYCALC
# ============================================================
print("\n📊 Loading FantasyCalc values...")
fc_url = "https://api.fantasycalc.com/values/current?isDynasty=true&numQbs=2&numTeams=12&ppr=1"
fc_data = requests.get(fc_url).json()

fc_map = {}  # name_lower → data
for item in fc_data:
    p = item.get('player', {})
    if not p or p.get('position') == 'PICK' or not p.get('sleeperId'):
        continue
    if p['position'] not in ('QB', 'RB', 'WR', 'TE'):
        continue
    val = item.get('value', 0)
    if val <= 0:
        continue
    trend = item.get('trend30Day', 0)
    fc_map[p['name'].lower().strip()] = {
        'sleeper_id': p['sleeperId'],
        'name': p['name'],
        'pos': p['position'],
        'team': p.get('maybeTeam', ''),
        'age': round(p.get('maybeAge', 0), 1),
        'yoe': p.get('maybeYoe', 0),
        'value': val,
        'trend': trend,
        'trendPct': round(trend / max(val, 1) * 100, 2),
        'rank': item.get('overallRank', 999),
        'posRank': item.get('positionRank', 999),
    }
print(f"   {len(fc_map)} players")

# ============================================================
# 2. LOAD NFLFASTR WEEKLY STATS
# ============================================================
print("\n🏈 Loading nflfastr weekly stats...")
current_year = datetime.now().year
seasons = [current_year - 2, current_year - 1, current_year]
seasons = [s for s in seasons if s >= 2022]

try:
    weekly = nfl.import_weekly_data(seasons)
    print(f"   {len(weekly)} player-weeks from {seasons}")
except Exception as e:
    print(f"   Warning: Could not load {seasons}, trying fallback...")
    seasons = [s for s in seasons if s < current_year]
    weekly = nfl.import_weekly_data(seasons)
    print(f"   {len(weekly)} player-weeks from {seasons}")

# ============================================================
# 3. BUILD FEATURES
# ============================================================
print("\n⚙️  Building features...")

def build_features(df, window=4):
    positions = ['QB', 'RB', 'WR', 'TE']
    df = df[df['position'].isin(positions)].copy()
    df = df.sort_values(['player_id', 'season', 'week'])
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    rows = []
    for pid, grp in df.groupby('player_id'):
        if len(grp) < window + 2:
            continue
        grp = grp.sort_values(['season', 'week']).reset_index(drop=True)
        for i in range(window, len(grp)):
            recent = grp.iloc[i - window:i]
            older = grp.iloc[max(0, i - window * 2):i - window] if i >= window * 2 else pd.DataFrame()
            cur = grp.iloc[i]

            row = {
                'player_id': pid,
                'player_name': cur.get('player_display_name', ''),
                'position': cur.get('position', ''),
                'season': int(cur['season']),
                'week': int(cur['week']),
            }

            for col in ['targets', 'receptions', 'carries', 'rushing_yards', 'receiving_yards',
                         'passing_yards', 'fantasy_points_ppr', 'target_share']:
                if col in recent.columns:
                    row[f'{col}_avg'] = round(recent[col].mean(), 2)

            if len(older) >= window // 2:
                for col in ['targets', 'carries', 'fantasy_points_ppr', 'target_share']:
                    if col in recent.columns and col in older.columns:
                        r_avg = recent[col].mean()
                        o_avg = older[col].mean()
                        row[f'{col}_trend'] = round(r_avg - o_avg, 2)
                        if o_avg > 0:
                            row[f'{col}_trend_pct'] = round((r_avg - o_avg) / o_avg * 100, 1)

            if 'targets' in recent.columns and 'carries' in recent.columns:
                row['touches_avg'] = round(recent['targets'].mean() + recent['carries'].mean(), 2)
            if 'receiving_yards' in recent.columns and 'targets' in recent.columns:
                t = recent['targets'].sum()
                if t > 0:
                    row['yards_per_target'] = round(recent['receiving_yards'].sum() / t, 2)
            if 'rushing_yards' in recent.columns and 'carries' in recent.columns:
                c = recent['carries'].sum()
                if c > 0:
                    row['yards_per_carry'] = round(recent['rushing_yards'].sum() / c, 2)

            rows.append(row)

    return pd.DataFrame(rows)

df_feat = build_features(weekly)
print(f"   {len(df_feat)} feature rows")

# Get latest week
latest = df_feat[df_feat['season'] == df_feat['season'].max()]
latest = latest[latest['week'] == latest['week'].max()].copy()
latest['merge_name'] = latest['player_name'].str.lower().str.strip()
print(f"   {len(latest)} players at latest week")

# ============================================================
# 4. MERGE + TRAIN + PREDICT
# ============================================================
print("\n🤖 Training models...")

fc_df = pd.DataFrame(fc_map.values())
fc_df['merge_name'] = fc_df['name'].str.lower().str.strip()

merged = latest.merge(fc_df[['merge_name', 'sleeper_id', 'value', 'trend', 'trendPct', 'age', 'yoe', 'rank', 'posRank']],
                       on='merge_name', how='inner')
print(f"   {len(merged)} players matched")

feat_cols = [c for c in merged.columns
             if c.endswith('_avg') or c.endswith('_trend') or c.endswith('_trend_pct')
             or c in ['age', 'yoe', 'value', 'rank', 'touches_avg', 'yards_per_target', 'yards_per_carry']]
valid_feats = [c for c in feat_cols if merged[c].notna().sum() > len(merged) * 0.3]

output = {}
model_info = {}

for pos in ['QB', 'RB', 'WR', 'TE']:
    pos_data = merged[merged['position'] == pos].copy()
    if len(pos_data) < 10:
        print(f"   {pos}: skipping ({len(pos_data)} players)")
        continue

    X = pos_data[valid_feats].fillna(0)
    y = pos_data['trend']

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                               min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X, y)

    pos_data['predicted'] = model.predict(X).round(0).astype(int)

    importances = pd.Series(model.feature_importances_, index=valid_feats).sort_values(ascending=False)
    mae = mean_absolute_error(y, model.predict(X))

    model_info[pos] = {
        'mae': round(mae),
        'n_players': len(pos_data),
        'top_features': importances.head(5).index.tolist()
    }
    print(f"   {pos}: {len(pos_data)} players, MAE={mae:.0f}")

    # Generate signals using both model prediction and age-based heuristics
    for _, row in pos_data.iterrows():
        sid = row['sleeper_id']
        pred = int(row['predicted'])
        age = row['age']
        val = row['value']
        trend_pct = row['trendPct']

        peak_age = {'QB': 32, 'RB': 27, 'WR': 29, 'TE': 29}.get(pos, 28)
        is_young = 0 < age < peak_age - 2
        is_old = age >= peak_age

        if is_old and (pred > 50 or trend_pct > 3) and val >= 3000:
            signal = 'SELL'
        elif is_young and (pred < -50 or trend_pct < -5) and val >= 1500:
            signal = 'BUY'
        elif is_young and (pred > 50 or trend_pct > 3):
            signal = 'HOLD+'
        elif pred > 150:
            signal = 'BUY'
        elif pred < -150:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # Build feature snapshot for the web tool
        feat_snapshot = {}
        for f in ['targets_avg', 'carries_avg', 'fantasy_points_ppr_avg', 'target_share_avg',
                    'targets_trend', 'carries_trend', 'target_share_trend', 'touches_avg',
                    'yards_per_target', 'yards_per_carry']:
            if f in row.index and pd.notna(row[f]):
                feat_snapshot[f] = round(float(row[f]), 2)

        output[sid] = {
            'name': row['player_name'],
            'pos': pos,
            'team': fc_map.get(row['merge_name'], {}).get('team', ''),
            'age': round(age, 1),
            'value': int(val),
            'trend': int(row['trend']),
            'trendPct': round(float(row['trendPct']), 1),
            'predicted': pred,
            'signal': signal,
            'features': feat_snapshot
        }

# ============================================================
# 5. WRITE OUTPUT
# ============================================================
print("\n💾 Writing data/predictions.json...")

result = {
    'updated': datetime.now(timezone.utc).isoformat(),
    'model_info': model_info,
    'players': output
}

os.makedirs('data', exist_ok=True)
with open('data/predictions.json', 'w') as f:
    json.dump(result, f, indent=2)

total_signals = sum(1 for p in output.values() if p['signal'] != 'HOLD')
print(f"   {len(output)} players, {total_signals} actionable signals")
print(f"   Signals: { {s: sum(1 for p in output.values() if p['signal']==s) for s in ['BUY','SELL','HOLD+','HOLD']} }")
print("\n✅ Done! data/predictions.json updated.")
