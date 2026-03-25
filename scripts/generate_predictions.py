"""
Dynasty Prediction Generator (GitHub Actions)
Outputs data/predictions.json for the web tool.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

import nfl_data_py as nfl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

print("=" * 60)
print("DYNASTY PREDICTION GENERATOR")
print(f"Run time: {datetime.now(timezone.utc).isoformat()}")
print("=" * 60)

# ============================================================
# SIGNAL LOGIC (one function, used everywhere)
# ============================================================
# BUY LOW  = young player whose value is DROPPING (buy the dip)
# SELL HIGH = older player whose value is RISING or at peak (sell before cliff)
# HOLD+    = young player whose value is rising (keep — great asset)
# AVOID    = old player whose value is dropping (don't buy)
# HOLD     = neutral / nothing actionable

def compute_signal(pos, age, value, trend, trend_pct):
    """Compute buy/sell signal based on age curve + value trend direction."""
    peak_age = {'QB': 32, 'RB': 27, 'WR': 29, 'TE': 29}.get(pos, 28)
    is_young = 0 < age < peak_age - 2
    is_prime = peak_age - 2 <= age < peak_age
    is_old = age >= peak_age
    is_rising = trend_pct > 3
    is_falling = trend_pct < -3
    is_high_val = value >= 3000
    is_mid_val = value >= 1500

    # SELL HIGH: value going UP but player is aging — peak before the cliff
    if is_old and is_rising and is_high_val:
        return 'SELL'
    # SELL HIGH: prime age, surging value — could be peaking
    if is_prime and trend_pct > 8 and is_high_val:
        return 'SELL'
    # AVOID: old and falling — stay away
    if is_old and is_falling and is_high_val:
        return 'AVOID'

    # BUY LOW: young player whose value is DROPPING — market overreaction
    if is_young and is_falling and is_mid_val:
        return 'BUY'
    # BUY LOW: prime player with significant drop — potential bargain
    if is_prime and trend_pct < -8 and is_high_val:
        return 'BUY'

    # HOLD+: young and rising — great dynasty asset, keep
    if is_young and is_rising:
        return 'HOLD+'
    # HOLD+: young and stable with high value — core piece
    if is_young and value >= 5000 and abs(trend_pct) < 3:
        return 'HOLD+'

    return 'HOLD'


# ============================================================
# 1. LOAD FANTASYCALC
# ============================================================
print("\n📊 Loading FantasyCalc values...")
fc_url = "https://api.fantasycalc.com/values/current?isDynasty=true&numQbs=2&numTeams=12&ppr=1"
fc_data = requests.get(fc_url).json()

fc_map = {}
for item in fc_data:
    p = item.get('player', {})
    if not p or p.get('position') == 'PICK' or not p.get('sleeperId'):
        continue
    if p['position'] not in ('QB', 'RB', 'WR', 'TE'):
        continue
    val = item.get('value', 0)
    if val <= 0:
        continue
    trend = item.get('trend30Day') or 0
    age = round((p.get('maybeAge') or 0), 1)
    trendPct = round(trend / max(val, 1) * 100, 2)
    pos = p['position']
    fc_map[p['name'].lower().strip()] = {
        'sleeper_id': p['sleeperId'],
        'name': p['name'],
        'pos': pos,
        'team': p.get('maybeTeam') or '',
        'age': age,
        'yoe': p.get('maybeYoe') or 0,
        'value': val,
        'trend': trend,
        'trendPct': trendPct,
        'rank': item.get('overallRank') or 999,
        'posRank': item.get('positionRank') or 999,
        'signal': compute_signal(pos, age, val, trend, trendPct),
    }
print(f"   {len(fc_map)} players")

# ============================================================
# 2. LOAD NFLFASTR WEEKLY STATS
# ============================================================
print("\n🏈 Loading nflfastr weekly stats...")
available_seasons = []
for yr in [2025, 2024, 2023, 2022]:
    try:
        test = nfl.import_weekly_data([yr])
        if len(test) > 0:
            available_seasons.append(yr)
            print(f"   ✓ {yr}: {len(test)} rows")
        if len(available_seasons) >= 3:
            break
    except Exception:
        print(f"   ✗ {yr}: not available")

if available_seasons:
    weekly = nfl.import_weekly_data(available_seasons)
    print(f"   Total: {len(weekly)} player-weeks from {available_seasons}")
else:
    weekly = pd.DataFrame()
    print("   No seasons available!")

# ============================================================
# 3. FEATURE BUILDER
# ============================================================
def build_features(df, window=4):
    df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()
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
            older = grp.iloc[max(0, i - window*2):i - window] if i >= window*2 else pd.DataFrame()
            cur = grp.iloc[i]
            row = {
                'player_id': pid,
                'player_name': cur.get('player_display_name', ''),
                'position': cur.get('position', ''),
                'season': int(cur['season']),
                'week': int(cur['week']),
            }
            for col in ['targets', 'receptions', 'carries', 'rushing_yards',
                        'receiving_yards', 'passing_yards', 'fantasy_points_ppr', 'target_share']:
                if col in recent.columns:
                    row[f'{col}_avg'] = round(recent[col].mean(), 2)
            if len(older) >= window // 2:
                for col in ['targets', 'carries', 'fantasy_points_ppr', 'target_share']:
                    if col in recent.columns and col in older.columns:
                        r_avg, o_avg = recent[col].mean(), older[col].mean()
                        row[f'{col}_trend'] = round(r_avg - o_avg, 2)
                        if o_avg > 0:
                            row[f'{col}_trend_pct'] = round((r_avg - o_avg) / o_avg * 100, 1)
            if 'targets' in recent.columns and 'carries' in recent.columns:
                row['touches_avg'] = round(recent['targets'].mean() + recent['carries'].mean(), 2)
            if 'receiving_yards' in recent.columns and 'targets' in recent.columns:
                t = recent['targets'].sum()
                if t > 0: row['yards_per_target'] = round(recent['receiving_yards'].sum() / t, 2)
            if 'rushing_yards' in recent.columns and 'carries' in recent.columns:
                c = recent['carries'].sum()
                if c > 0: row['yards_per_carry'] = round(recent['rushing_yards'].sum() / c, 2)
            rows.append(row)
    return pd.DataFrame(rows)

# ============================================================
# 4. GENERATE PREDICTIONS
# ============================================================
output = {}
model_info = {}

# Helper: normalize name for matching
def norm_name(n):
    return n.lower().strip().replace('.', '').replace('-', '').replace("'", '').replace(' jr', '').replace(' iii', '').replace(' ii', '').replace(' iv', '')

# TRY MODEL PATH (nflfastr + XGBoost)
if len(weekly) > 0:
    print("\n⚙️  Building features...")
    df_feat = build_features(weekly)
    print(f"   {len(df_feat)} feature rows")

    latest = df_feat[df_feat['season'] == df_feat['season'].max()]
    latest = latest[latest['week'] == latest['week'].max()].copy()
    latest['merge_name'] = latest['player_name'].apply(norm_name)
    print(f"   {len(latest)} players at latest week")
    print(f"   nflfastr samples: {latest['merge_name'].head(5).tolist()}")

    print("\n🤖 Training models...")
    fc_df = pd.DataFrame(fc_map.values())
    fc_df['merge_name'] = fc_df['name'].apply(norm_name)
    print(f"   FantasyCalc samples: {fc_df['merge_name'].head(5).tolist()}")

    merged = latest.merge(
        fc_df[['merge_name', 'sleeper_id', 'value', 'trend', 'trendPct', 'age', 'yoe', 'rank', 'posRank', 'signal']],
        on='merge_name', how='inner'
    )
    print(f"   {len(merged)} players matched")

    if len(merged) < 50:
        fc_names = set(fc_df['merge_name'].tolist())
        nfl_names = set(latest['merge_name'].tolist())
        print(f"   ⚠️ Low match! FC-only samples: {list(fc_names - nfl_names)[:8]}")
        print(f"   ⚠️ nflfastr-only samples: {list(nfl_names - fc_names)[:8]}")

    if len(merged) >= 50:
        feat_cols = [c for c in merged.columns
                     if c.endswith('_avg') or c.endswith('_trend') or c.endswith('_trend_pct')
                     or c in ['age', 'yoe', 'value', 'rank', 'touches_avg', 'yards_per_target', 'yards_per_carry']]
        valid_feats = [c for c in feat_cols if merged[c].notna().sum() > len(merged) * 0.3]

        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_data = merged[merged['position'] == pos].copy()
            if len(pos_data) < 10:
                print(f"   {pos}: skipping ({len(pos_data)} players)")
                continue

            X = pos_data[valid_feats].fillna(0)
            y = pos_data['trend']

            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42
            )
            model.fit(X, y)
            pos_data = pos_data.copy()
            pos_data['predicted'] = model.predict(X).round(0).astype(int)

            importances = pd.Series(model.feature_importances_, index=valid_feats).sort_values(ascending=False)
            mae = mean_absolute_error(y, model.predict(X))
            model_info[pos] = {
                'mae': round(mae),
                'n_players': len(pos_data),
                'top_features': importances.head(5).index.tolist()
            }
            print(f"   {pos}: {len(pos_data)} players, MAE={mae:.0f}, top={importances.index[0]}")

            for _, row in pos_data.iterrows():
                sid = row['sleeper_id']
                pred = int(row['predicted'])
                age = float(row['age'] or 0)
                val = int(row['value'] or 0)
                trend_val = int(row['trend'] if pd.notna(row['trend']) else 0)
                trend_pct = float(row['trendPct'] if pd.notna(row['trendPct']) else 0)

                # Use the proper signal function
                signal = compute_signal(pos, age, val, trend_val, trend_pct)

                feat_snapshot = {}
                for f in ['targets_avg', 'carries_avg', 'fantasy_points_ppr_avg', 'target_share_avg',
                           'targets_trend', 'carries_trend', 'target_share_trend', 'touches_avg',
                           'yards_per_target', 'yards_per_carry']:
                    if f in row.index and pd.notna(row[f]):
                        feat_snapshot[f] = round(float(row[f]), 2)

                output[sid] = {
                    'name': row['player_name'], 'pos': pos,
                    'team': fc_map.get(row['merge_name'], {}).get('team', ''),
                    'age': round(age, 1), 'value': val,
                    'trend': trend_val, 'trendPct': round(trend_pct, 1),
                    'predicted': pred, 'signal': signal,
                    'features': feat_snapshot
                }

# ============================================================
# 5. SAFETY NET — if model produced nothing, use FC-only
# ============================================================
if len(output) == 0:
    if len(weekly) > 0:
        print("\n⚠️  Model merge found too few matches — falling back to FantasyCalc trends...")
    else:
        print("\n⚙️  No weekly stats — using FantasyCalc trend-only predictions...")

    for name_key, pdata in fc_map.items():
        output[pdata['sleeper_id']] = {
            'name': pdata['name'],
            'pos': pdata['pos'],
            'team': pdata['team'],
            'age': pdata['age'],
            'value': pdata['value'],
            'trend': pdata['trend'],
            'trendPct': pdata['trendPct'],
            'predicted': pdata['trend'],
            'signal': pdata['signal'],  # already computed with compute_signal()
            'features': {
                'rank': pdata['rank'],
                'pos_rank': pdata['posRank'],
                'trend_30d': pdata['trend'],
                'trend_pct': pdata['trendPct'],
                'age': pdata['age'],
                'years_exp': pdata['yoe'],
            }
        }
    model_info = {'note': 'FantasyCalc trends + age curves (no nflfastr model this run)'}
    print(f"   Generated {len(output)} predictions")

# ============================================================
# 6. WRITE OUTPUT
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

signal_counts = {}
for p in output.values():
    s = p['signal']
    signal_counts[s] = signal_counts.get(s, 0) + 1
print(f"   {len(output)} players")
print(f"   Signals: {signal_counts}")

# Print a few examples for verification
print("\n📋 Sample outputs for verification:")
for name_key in ['josh allen', 'saquon barkley', 'jamarr chase', 'bijan robinson', 'sam laporte']:
    if name_key in fc_map:
        sid = fc_map[name_key]['sleeper_id']
        if sid in output:
            p = output[sid]
            print(f"   {p['name']:25s}  {p['pos']}  Age {p['age']:4.1f}  Val {p['value']:>6,}  "
                  f"Trend {p['trend']:>+5}  ({p['trendPct']:>+5.1f}%)  → {p['signal']}")

print("\n✅ Done!")
