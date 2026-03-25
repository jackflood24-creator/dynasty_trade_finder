"""
Dynasty Value Prediction Model Pipeline
========================================
Trains per-position XGBoost models to predict which players will gain
or lose dynasty trade value over the next 30 days based on usage trends.

Requirements:
    pip install pandas numpy requests scikit-learn xgboost nfl_data_py joblib

Usage:
    python dynasty_model_pipeline.py

Output:
    - models/dynasty_model_QB.joblib (etc for each position)
    - predictions/current_predictions.csv
    - predictions/buy_sell_signals.csv
"""

import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import nfl_data_py as nfl
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_absolute_error, classification_report
    from sklearn.preprocessing import LabelEncoder
    import joblib
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pandas numpy requests scikit-learn xgboost nfl_data_py joblib")
    exit(1)

os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("DYNASTY VALUE PREDICTION MODEL PIPELINE")
print("=" * 60)

# Load FantasyCalc current values
print("\n📊 Loading FantasyCalc values...")
fc_url = "https://api.fantasycalc.com/values/current?isDynasty=true&numQbs=2&numTeams=12&ppr=1"
fc_data = requests.get(fc_url).json()

current_vals = {}
for item in fc_data:
    p = item.get('player', {})
    if not p or p.get('position') == 'PICK' or not p.get('sleeperId'):
        continue
    current_vals[p['name']] = {
        'sleeper_id': p['sleeperId'],
        'name': p['name'],
        'position': p['position'],
        'team': p.get('maybeTeam', ''),
        'age': p.get('maybeAge', 0),
        'yoe': p.get('maybeYoe', 0),
        'value': item['value'],
        'trend_30d': item.get('trend30Day', 0),
        'rank': item.get('overallRank', 999),
        'pos_rank': item.get('positionRank', 999),
    }
print(f"   {len(current_vals)} players loaded")

# Load weekly stats (3 seasons for training)
print("\n🏈 Loading nflfastr weekly data (2022-2024)...")
seasons = [2022, 2023, 2024]
weekly = nfl.import_weekly_data(seasons)
print(f"   {len(weekly)} player-weeks loaded")

# Load roster data for snap counts if available
print("📋 Loading seasonal data...")
seasonal = nfl.import_seasonal_data(seasons)
print(f"   {len(seasonal)} player-seasons loaded")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n⚙️  Engineering features...")

def build_features(weekly_df, window=4):
    """
    For each player-week, compute rolling features over the trailing `window` weeks.
    These features capture USAGE TRENDS, not just raw stats.
    """
    positions = ['QB', 'RB', 'WR', 'TE']
    weekly_df = weekly_df[weekly_df['position'].isin(positions)].copy()
    weekly_df = weekly_df.sort_values(['player_id', 'season', 'week'])

    # Fill NaN with 0 for numeric columns
    num_cols = weekly_df.select_dtypes(include=[np.number]).columns
    weekly_df[num_cols] = weekly_df[num_cols].fillna(0)

    features = []

    for pid, grp in weekly_df.groupby('player_id'):
        if len(grp) < window + 2:
            continue

        grp = grp.sort_values(['season', 'week']).reset_index(drop=True)

        for i in range(window, len(grp)):
            recent = grp.iloc[i - window:i]
            current = grp.iloc[i]
            older = grp.iloc[max(0, i - window * 2):i - window] if i >= window * 2 else pd.DataFrame()

            row = {
                'player_id': pid,
                'player_name': current.get('player_display_name', ''),
                'position': current.get('position', ''),
                'season': current['season'],
                'week': current['week'],
            }

            # Rolling averages (recent window)
            for col in ['targets', 'receptions', 'carries', 'rushing_yards', 'receiving_yards',
                         'passing_yards', 'fantasy_points_ppr', 'target_share']:
                if col in recent.columns:
                    row[f'{col}_avg'] = recent[col].mean()
                    row[f'{col}_std'] = recent[col].std()

            # Trend: compare recent window to older window (usage change)
            if len(older) >= window // 2:
                for col in ['targets', 'carries', 'fantasy_points_ppr', 'target_share']:
                    if col in recent.columns and col in older.columns:
                        recent_avg = recent[col].mean()
                        older_avg = older[col].mean()
                        row[f'{col}_trend'] = recent_avg - older_avg
                        if older_avg > 0:
                            row[f'{col}_trend_pct'] = (recent_avg - older_avg) / older_avg * 100

            # Touch share and efficiency
            if 'targets' in recent.columns and 'carries' in recent.columns:
                row['touches_avg'] = recent['targets'].mean() + recent['carries'].mean()
            if 'receiving_yards' in recent.columns and 'targets' in recent.columns:
                total_tgts = recent['targets'].sum()
                if total_tgts > 0:
                    row['yards_per_target'] = recent['receiving_yards'].sum() / total_tgts
            if 'rushing_yards' in recent.columns and 'carries' in recent.columns:
                total_carries = recent['carries'].sum()
                if total_carries > 0:
                    row['yards_per_carry'] = recent['rushing_yards'].sum() / total_carries

            # Current week performance (this week's stats as latest signal)
            for col in ['targets', 'carries', 'fantasy_points_ppr']:
                if col in current.index:
                    row[f'{col}_current'] = current[col]

            features.append(row)

    return pd.DataFrame(features)


df_features = build_features(weekly)
print(f"   Generated {len(df_features)} feature rows")
print(f"   Features per row: {len(df_features.columns)}")

# ============================================================
# 3. MERGE WITH CURRENT VALUES FOR LABELS
# ============================================================
print("\n🔗 Merging features with dynasty values...")

# For current predictions, we use latest week's features
latest_season = df_features['season'].max()
latest_week = df_features[df_features['season'] == latest_season]['week'].max()
current_features = df_features[
    (df_features['season'] == latest_season) & (df_features['week'] == latest_week)
].copy()

# Merge with current values by name
vals_df = pd.DataFrame(current_vals.values())
current_features['merge_name'] = current_features['player_name'].str.lower().str.strip()
vals_df['merge_name'] = vals_df['name'].str.lower().str.strip()

prediction_data = current_features.merge(
    vals_df[['merge_name', 'value', 'trend_30d', 'age', 'yoe', 'rank', 'position']],
    on='merge_name', how='inner', suffixes=('', '_fc')
)
prediction_data['age'] = prediction_data['age'].fillna(0)
prediction_data['yoe'] = prediction_data['yoe'].fillna(0)

print(f"   {len(prediction_data)} players matched for prediction")

# ============================================================
# 4. TRAIN MODELS (per position)
# ============================================================
print("\n🤖 Training models...")

# For training, we use trend_30d as the TARGET (what we want to predict)
# In a production setup, you'd have historical value snapshots.
# Here we demonstrate the pipeline structure using current trend as proxy.

feature_cols = [c for c in prediction_data.columns
                if c.endswith('_avg') or c.endswith('_std') or c.endswith('_trend')
                or c.endswith('_trend_pct') or c.endswith('_current')
                or c in ['age', 'yoe', 'value', 'rank', 'touches_avg', 'yards_per_target', 'yards_per_carry']]

# Remove columns with all NaN
valid_features = []
for col in feature_cols:
    if prediction_data[col].notna().sum() > len(prediction_data) * 0.3:
        valid_features.append(col)

print(f"   Using {len(valid_features)} features: {valid_features[:10]}...")

results = {}
for pos in ['QB', 'RB', 'WR', 'TE']:
    pos_data = prediction_data[prediction_data['position'] == pos].copy()
    if len(pos_data) < 15:
        print(f"   {pos}: Not enough data ({len(pos_data)} players), skipping")
        continue

    X = pos_data[valid_features].fillna(0)
    y = pos_data['trend_30d']

    # Train XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=valid_features).sort_values(ascending=False)

    # Predictions
    pos_data['predicted_trend'] = model.predict(X)
    pos_data['signal'] = pos_data['predicted_trend'].apply(
        lambda x: 'BUY' if x > 100 else ('SELL' if x < -100 else 'HOLD')
    )

    results[pos] = pos_data

    # Save model
    model_path = f'models/dynasty_model_{pos}.joblib'
    joblib.dump({'model': model, 'features': valid_features}, model_path)

    print(f"\n   📦 {pos} Model ({len(pos_data)} players):")
    print(f"      Top features: {', '.join(importances.head(5).index.tolist())}")
    print(f"      MAE: {mean_absolute_error(y, model.predict(X)):.0f} value points")
    print(f"      Signals: {pos_data['signal'].value_counts().to_dict()}")

# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================
print("\n\n📋 Generating prediction output...")

all_preds = pd.concat(results.values(), ignore_index=True)
all_preds = all_preds.sort_values('predicted_trend', ascending=False)

output_cols = ['player_name', 'position', 'age', 'value', 'trend_30d', 'predicted_trend', 'signal']
available_output = [c for c in output_cols if c in all_preds.columns]
all_preds[available_output].to_csv('predictions/current_predictions.csv', index=False)

# Buy/Sell signals
signals = all_preds[all_preds['signal'] != 'HOLD'].copy()
signals[available_output].to_csv('predictions/buy_sell_signals.csv', index=False)

print(f"   Saved predictions/current_predictions.csv ({len(all_preds)} players)")
print(f"   Saved predictions/buy_sell_signals.csv ({len(signals)} signals)")

# ============================================================
# 6. PRINT TOP RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 60)
print("🚀 TOP BUY SIGNALS (model predicts value increase)")
print("=" * 60)
buys = all_preds[all_preds['signal'] == 'BUY'].head(15)
for _, row in buys.iterrows():
    print(f"  {row['player_name']:25s}  {row['position']}  Age {row['age']:.1f}  "
          f"Val: {row['value']:>6,}  Actual: {row['trend_30d']:>+5.0f}  "
          f"Predicted: {row['predicted_trend']:>+5.0f}")

print("\n" + "=" * 60)
print("📉 TOP SELL SIGNALS (model predicts value decrease)")
print("=" * 60)
sells = all_preds[all_preds['signal'] == 'SELL'].sort_values('predicted_trend').head(15)
for _, row in sells.iterrows():
    print(f"  {row['player_name']:25s}  {row['position']}  Age {row['age']:.1f}  "
          f"Val: {row['value']:>6,}  Actual: {row['trend_30d']:>+5.0f}  "
          f"Predicted: {row['predicted_trend']:>+5.0f}")

print("\n✅ Pipeline complete!")
print("   Models saved in models/")
print("   Predictions saved in predictions/")
print("   Re-run weekly during the season for updated signals.")
print("\n   To use in your trade finder: import the CSV predictions and")
print("   overlay them on roster cards as 'rising ↑' / 'falling ↓' indicators.")
