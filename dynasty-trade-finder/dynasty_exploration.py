"""
Dynasty Value Momentum — Data Exploration
==========================================
This script explores the relationship between NFL player usage stats
and dynasty trade value changes. Run this first to understand the data
before building the predictive model.

Requirements:
    pip install pandas numpy requests matplotlib seaborn nfl_data_py

Data sources:
    - nfl_data_py: play-by-play and weekly stats from nflfastr
    - FantasyCalc API: dynasty trade values with 30-day trends
"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import nfl_data_py as nfl
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
sns.set_palette('Set2')

# ============================================================
# 1. LOAD FANTASYCALC CURRENT VALUES
# ============================================================
print("📊 Loading FantasyCalc dynasty values...")

fc_url = "https://api.fantasycalc.com/values/current?isDynasty=true&numQbs=2&numTeams=12&ppr=1"
fc_data = requests.get(fc_url).json()

fc_players = []
for item in fc_data:
    p = item.get('player', {})
    if not p or p.get('position') == 'PICK':
        continue
    fc_players.append({
        'sleeper_id': p.get('sleeperId', ''),
        'name': p.get('name', ''),
        'position': p.get('position', ''),
        'team': p.get('maybeTeam', ''),
        'age': p.get('maybeAge', 0),
        'yoe': p.get('maybeYoe', 0),
        'value': item.get('value', 0),
        'trend_30d': item.get('trend30Day', 0),
        'overall_rank': item.get('overallRank', 999),
        'pos_rank': item.get('positionRank', 999),
        'tier': item.get('maybeTier', 0),
        'std_dev': item.get('maybeMovingStandardDeviation', 0),
    })

df_vals = pd.DataFrame(fc_players)
df_vals['trend_pct'] = (df_vals['trend_30d'] / df_vals['value'].clip(lower=1) * 100).round(2)
df_vals = df_vals[df_vals['position'].isin(['QB', 'RB', 'WR', 'TE']) & (df_vals['value'] > 0)]

print(f"   Loaded {len(df_vals)} players with values")
print(f"   Positions: {df_vals['position'].value_counts().to_dict()}")
print()

# ============================================================
# 2. LOAD NFLFASTR WEEKLY STATS (Last 2 seasons)
# ============================================================
print("🏈 Loading weekly stats from nflfastr...")

seasons = [2023, 2024]
weekly = nfl.import_weekly_data(seasons)

# Key columns for dynasty value prediction
stat_cols = [
    'player_id', 'player_name', 'player_display_name', 'position', 'season', 'week',
    'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
    'carries', 'rushing_yards', 'rushing_tds',
    'receptions', 'targets', 'receiving_yards', 'receiving_tds',
    'target_share', 'air_yards_share', 'wopr',
    'fantasy_points', 'fantasy_points_ppr'
]
available_cols = [c for c in stat_cols if c in weekly.columns]
weekly = weekly[available_cols].copy()

print(f"   Loaded {len(weekly)} player-weeks across {seasons}")
print()

# ============================================================
# 3. COMPUTE USAGE METRICS
# ============================================================
print("⚙️ Computing usage metrics...")

# Per-player season aggregates
def compute_player_stats(df):
    """Compute per-player, per-season summary stats."""
    stats = df.groupby(['player_id', 'player_display_name', 'position', 'season']).agg(
        games=('week', 'nunique'),
        total_targets=('targets', 'sum'),
        total_receptions=('receptions', 'sum'),
        total_carries=('carries', 'sum'),
        total_rush_yds=('rushing_yards', 'sum'),
        total_rec_yds=('receiving_yards', 'sum'),
        total_pass_yds=('passing_yards', 'sum'),
        total_tds=('rushing_tds', lambda x: x.sum()),
        total_rec_tds=('receiving_tds', 'sum'),
        total_pass_tds=('passing_tds', 'sum'),
        avg_target_share=('target_share', 'mean'),
        avg_fantasy_ppr=('fantasy_points_ppr', 'mean'),
        total_fantasy_ppr=('fantasy_points_ppr', 'sum'),
    ).reset_index()

    # Per-game rates
    stats['targets_per_game'] = stats['total_targets'] / stats['games'].clip(lower=1)
    stats['carries_per_game'] = stats['total_carries'] / stats['games'].clip(lower=1)
    stats['touches_per_game'] = (stats['total_targets'] + stats['total_carries']) / stats['games'].clip(lower=1)
    stats['yards_per_game'] = (stats['total_rush_yds'] + stats['total_rec_yds'] + stats['total_pass_yds']) / stats['games'].clip(lower=1)

    return stats

player_stats = compute_player_stats(weekly)
print(f"   Computed stats for {len(player_stats)} player-seasons")
print()

# ============================================================
# 4. MERGE STATS WITH DYNASTY VALUES
# ============================================================
print("🔗 Merging stats with dynasty values...")

# nflfastr uses gsis_id, FantasyCalc uses sleeper_id — need to bridge via name+team
# Simple merge on name (imperfect but works for exploration)
df_vals['merge_name'] = df_vals['name'].str.lower().str.strip()
latest_stats = player_stats[player_stats['season'] == player_stats['season'].max()].copy()
latest_stats['merge_name'] = latest_stats['player_display_name'].str.lower().str.strip()

merged = df_vals.merge(latest_stats, on='merge_name', how='inner', suffixes=('', '_stats'))
print(f"   Merged {len(merged)} players (out of {len(df_vals)} with values)")
print()

# ============================================================
# 5. EXPLORE: VALUE vs AGE by POSITION
# ============================================================
print("📈 Generating exploration plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dynasty Value vs Age by Position', fontsize=16, fontweight='bold', color='white')

for idx, pos in enumerate(['QB', 'RB', 'WR', 'TE']):
    ax = axes[idx // 2][idx % 2]
    pos_data = merged[merged['position'] == pos]
    if len(pos_data) == 0:
        continue

    scatter = ax.scatter(
        pos_data['age'], pos_data['value'],
        c=pos_data['trend_30d'], cmap='RdYlGn', s=40, alpha=0.7,
        edgecolors='white', linewidth=0.3, vmin=-500, vmax=500
    )
    ax.set_title(f'{pos} ({len(pos_data)} players)', fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Dynasty Value')

    # Label top players
    top = pos_data.nlargest(5, 'value')
    for _, row in top.iterrows():
        ax.annotate(row['name'], (row['age'], row['value']),
                     fontsize=7, color='white', alpha=0.8,
                     xytext=(5, 5), textcoords='offset points')

plt.colorbar(scatter, ax=axes, label='30-Day Trend', shrink=0.6)
plt.tight_layout()
plt.savefig('01_value_vs_age.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()
print("   Saved: 01_value_vs_age.png")

# ============================================================
# 6. EXPLORE: USAGE vs VALUE TREND
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Usage Metrics vs 30-Day Value Trend', fontsize=14, fontweight='bold', color='white')

skill = merged[merged['position'].isin(['RB', 'WR', 'TE'])]

metrics = [
    ('targets_per_game', 'Targets/Game'),
    ('touches_per_game', 'Touches/Game'),
    ('avg_fantasy_ppr', 'Avg PPR Pts/Game'),
]

for i, (col, label) in enumerate(metrics):
    if col not in skill.columns:
        continue
    ax = axes[i]
    ax.scatter(skill[col], skill['trend_pct'], alpha=0.5, s=30, c=skill['position'].map(
        {'RB': '#448aff', 'WR': '#00e676', 'TE': '#ffd740'}
    ), edgecolors='white', linewidth=0.3)
    ax.axhline(y=0, color='white', alpha=0.2, linestyle='--')
    ax.set_xlabel(label)
    ax.set_ylabel('30-Day Trend %')
    ax.set_title(label)

    # Add trend line
    valid = skill[[col, 'trend_pct']].dropna()
    if len(valid) > 10:
        z = np.polyfit(valid[col], valid['trend_pct'], 1)
        x_line = np.linspace(valid[col].min(), valid[col].max(), 50)
        ax.plot(x_line, z[0] * x_line + z[1], color='#ff5252', linewidth=2, alpha=0.8)

plt.tight_layout()
plt.savefig('02_usage_vs_trend.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()
print("   Saved: 02_usage_vs_trend.png")

# ============================================================
# 7. EXPLORE: AGE CURVES by POSITION
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Dynasty Value Age Curves by Position', fontsize=14, fontweight='bold', color='white')

colors = {'QB': '#ff5252', 'RB': '#448aff', 'WR': '#00e676', 'TE': '#ffd740'}
for pos in ['QB', 'RB', 'WR', 'TE']:
    pos_data = df_vals[df_vals['position'] == pos]
    if len(pos_data) < 10:
        continue
    age_curve = pos_data.groupby(pos_data['age'].round(0))['value'].median()
    age_curve = age_curve[age_curve.index.between(20, 38)]
    ax.plot(age_curve.index, age_curve.values, marker='o', markersize=4,
            linewidth=2, label=pos, color=colors.get(pos, 'white'))

ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Median Dynasty Value', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.15)
plt.tight_layout()
plt.savefig('03_age_curves.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.show()
print("   Saved: 03_age_curves.png")

# ============================================================
# 8. KEY CORRELATIONS
# ============================================================
print("\n📊 Correlation with 30-day trend (skill positions):")
print("=" * 50)
corr_cols = ['targets_per_game', 'carries_per_game', 'touches_per_game',
             'avg_fantasy_ppr', 'avg_target_share', 'yards_per_game',
             'age', 'yoe', 'value', 'std_dev']
available_corr = [c for c in corr_cols if c in merged.columns]
correlations = merged[merged['position'].isin(['RB', 'WR', 'TE'])][available_corr + ['trend_pct']].corr()['trend_pct'].drop('trend_pct').sort_values()
for col, val in correlations.items():
    bar = '█' * int(abs(val) * 40)
    sign = '+' if val > 0 else '-'
    print(f"  {col:25s}  {sign}{abs(val):.3f}  {bar}")

# ============================================================
# 9. SUMMARY TABLE: BIGGEST BUY-LOW / SELL-HIGH
# ============================================================
print("\n\n🎯 TOP BUY-LOW CANDIDATES (young, dropping value, strong usage):")
print("=" * 80)
buy_low = merged[
    (merged['age'] < 26) &
    (merged['trend_pct'] < -5) &
    (merged['value'] > 2000) &
    (merged['position'].isin(['RB', 'WR', 'TE']))
].sort_values('trend_pct').head(10)

for _, row in buy_low.iterrows():
    print(f"  {row['name']:25s}  {row['position']}  Age {row['age']:.1f}  "
          f"Val: {row['value']:,}  Trend: {row['trend_pct']:+.1f}%  "
          f"Tgts/g: {row.get('targets_per_game', 0):.1f}  PPR/g: {row.get('avg_fantasy_ppr', 0):.1f}")

print("\n🔥 TOP SELL-HIGH CANDIDATES (older, rising value):")
print("=" * 80)
sell_high = merged[
    (merged['age'] >= 28) &
    (merged['trend_pct'] > 5) &
    (merged['value'] > 3000) &
    (merged['position'].isin(['RB', 'WR', 'TE']))
].sort_values('trend_pct', ascending=False).head(10)

for _, row in sell_high.iterrows():
    print(f"  {row['name']:25s}  {row['position']}  Age {row['age']:.1f}  "
          f"Val: {row['value']:,}  Trend: {row['trend_pct']:+.1f}%  "
          f"Tgts/g: {row.get('targets_per_game', 0):.1f}  PPR/g: {row.get('avg_fantasy_ppr', 0):.1f}")

print("\n✅ Exploration complete! Check the PNG files for visualizations.")
print("   Next step: run dynasty_model_pipeline.py to train predictive models.")
