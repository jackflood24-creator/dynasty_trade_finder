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
    fc_map[p['name'].lower().strip()] = {
        'sleeper_id': p['sleeperId'],
        'name': p['name'],
        'pos': p['position'],
        'team': p.get('maybeTeam') or '',
        'age': round((p.get('maybeAge') or 0), 1),
        'yoe': p.get('maybeYoe') or 0,
        'value': val,
        'trend': trend,
        'trendPct': round(trend / max(val, 1) * 100, 2),
        'rank': item.get('overallRank') or 999,
        'posRank': item.get('positionRank') or 999,
    }
print(f"   {len(fc_map)} players")

# ============================================================
# 2. LOAD NFLFASTR WEEKLY STATS (try each year individually)
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

if len(weekly) > 0:
    print("\n⚙️  Building features...")
    df_feat = build_features(weekly)
    print(f"   {len(df_feat)} feature rows")

    latest = df_feat[df_feat['season'] == df_feat['season'].max()]
    latest = latest[latest['week'] == latest['week'].max()].copy()
    latest['merge_name'] = latest['player_name'].str.lower().str.strip()
    print(f"   {len(latest)} players at latest week")

    print("\n🤖 Training models...")
    fc_df = pd.DataFrame(fc_map.values())
    fc_df['merge_name'] = fc_df['name'].str.lower().str.strip()

    merged = latest.merge(
        fc_df[['merge_name', 'sleeper_id', 'value', 'trend', 'trendPct', 'age', 'yoe', 'rank', 'posRank']],
        on='merge_name', how='inner'
    )
    print(f"   {len(merged)} players matched")

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
        print(f"   {pos}: {len(pos_data)} players, MAE={mae:.0f}")

        for _, row in pos_data.iterrows():
            sid = row['sleeper_id']
            pred = int(row['predicted'])
            age = float(row['age'] or 0)
            val = int(row['value'] or 0)
            trend_pct = float(row['trendPct'] if pd.notna(row['trendPct']) else 0)

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
                'trend': int(row['trend'] if pd.notna(row['trend']) else 0),
                'trendPct': round(trend_pct, 1),
                'predicted': pred, 'signal': signal,
                'features': feat_snapshot
            }

else:
    # FALLBACK: FantasyCalc trends only (no nflfastr data)
    print("\n⚙️  No weekly stats — using FantasyCalc trend-only predictions...")
    for name_key, pdata in fc_map.items():
        age = pdata['age']
        val = pdata['value']
        trend = pdata['trend']
        trendPct = pdata['trendPct']
        pos = pdata['pos']

        peak_age = {'QB': 32, 'RB': 27, 'WR': 29, 'TE': 29}.get(pos, 28)
        is_young = 0 < age < peak_age - 2
        is_old = age >= peak_age

        if is_old and trendPct > 3 and val >= 3000: signal = 'SELL'
        elif is_young and trendPct < -5 and val >= 1500: signal = 'BUY'
        elif is_young and trendPct > 3: signal = 'HOLD+'
        elif trend > 200: signal = 'BUY'
        elif trend < -200: signal = 'SELL'
        else: signal = 'HOLD'

        output[pdata['sleeper_id']] = {
            'name': pdata['name'], 'pos': pos, 'team': pdata['team'],
            'age': age, 'value': val, 'trend': trend, 'trendPct': trendPct,
            'predicted': trend, 'signal': signal, 'features': {}
        }
    model_info = {'note': 'FantasyCalc trends only — no nflfastr data available'}
    print(f"   Generated {len(output)} trend-based predictions")

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
signal_counts = {}
for p in output.values():
    s = p['signal']
    signal_counts[s] = signal_counts.get(s, 0) + 1
print(f"   Signals: {signal_counts}")
print("\n✅ Done! data/predictions.json updated.")
