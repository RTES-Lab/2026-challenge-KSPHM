"""0514 results visualization (English only)"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

BASE = '/data/home/ksphm/2026-challenge-KSPHM/User/SR/0514'

# --- Load data ---
train_hi = {}
for i in range(1, 5):
    df = pd.read_csv(f'{BASE}/hi/output/train_v4/Bearing{i}_best.csv')
    train_hi[i] = df['HI'].values

test_hi = {}
for i in range(1, 7):
    df = pd.read_csv(f'{BASE}/hi/output/test_v4/Test{i}_best.csv')
    test_hi[i] = df['HI'].values

hi_summary_test  = pd.read_csv(f'{BASE}/hi/output/test_v4/summary_test_v4.csv')
hi_summary_train = pd.read_csv(f'{BASE}/hi/output/train_v4/summary_best.csv')
rul_summary      = pd.read_csv(f'{BASE}/rul/output/test/Test_RUL_summary.csv')
rul_all          = pd.read_csv(f'{BASE}/rul/output/test/All_Test_RUL_results.csv')

TRAIN_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
TEST_COLORS  = ['#F44336', '#9C27B0', '#00BCD4', '#FF5722', '#607D8B', '#795548']

v1_scores = [0.819, 0.133, 0.598, 0.594, 0.754, 0.592]
v3_scores = [0.5172, 0.3791, 0.6577, 0.6761, 0.7855, 0.6542]
v4_scores = list(hi_summary_test['q_score'])

# ===============================================================
# Figure 1: Summary Dashboard
# ===============================================================
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, top=0.95, bottom=0.04)
fig.suptitle('0514 Experiment Results — Summary Dashboard  [HI v4: Train-Anchored]', fontsize=22, fontweight='bold', y=0.98)

# -- Train HI curves (Bearing 1-4) --
loocv_scores = {1: 0.695, 2: 0.521, 3: 0.619, 4: 0.551}
train_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
              fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

for idx, (ax, b) in enumerate(zip(train_axes, range(1, 5))):
    hi = train_hi[b]
    x  = np.arange(len(hi))
    ax.plot(x, hi, color=TRAIN_COLORS[idx], lw=2, label=f'Bearing{b} HI')
    ax.axhline(0.8, color='red', ls='--', lw=1, alpha=0.7, label='Threshold 0.8')
    ax.fill_between(x, hi, alpha=0.15, color=TRAIN_COLORS[idx])
    ax.set_title(f'Bearing{b}  |  Life={len(hi)-1} cycles  |  LOOCV={loocv_scores[b]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Cycle', fontsize=9)
    ax.set_ylabel('Health Index', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.scatter([len(hi)-1], [hi[-1]], color='red', zorder=5, s=60)
    ax.annotate(f'HI={hi[-1]:.3f}', (len(hi)-1, hi[-1]),
                textcoords='offset points', xytext=(-40, 8),
                fontsize=8, color='red', fontweight='bold')

# -- HI Q-Score: v1 vs v3 --
ax_q   = fig.add_subplot(gs[0, 2])
x_pos  = np.arange(6)
width  = 0.35
width = 0.25
ax_q.bar(x_pos - width, v1_scores, width, label='v1 (K-Means+SignalTransform)',
          color='#90A4AE', alpha=0.8, edgecolor='gray')
ax_q.bar(x_pos,          v3_scores, width, label='v3 (FFT+FDR)',
          color='#42A5F5', alpha=0.9, edgecolor='#1565C0')
ax_q.bar(x_pos + width,  v4_scores, width, label='v4 (Train-Anchored)',
          color='#66BB6A', alpha=0.9, edgecolor='#2E7D32')

for i, (v3, v4, x) in enumerate(zip(v3_scores, v4_scores, x_pos)):
    delta = v4 - v3
    color = '#2E7D32' if delta > 0 else '#C62828'
    sign  = '+' if delta >= 0 else ''
    ax_q.text(x + width, max(v3, v4) + 0.02, f'{sign}{delta:.3f}',
              ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

ax_q.set_title('HI Q-Score: v1 / v3 / v4', fontsize=11, fontweight='bold')
ax_q.set_xticks(x_pos)
ax_q.set_xticklabels([f'Test{i}' for i in range(1, 7)], fontsize=9)
ax_q.set_ylabel('Q-Score', fontsize=9)
ax_q.set_ylim(0, 1.15)
ax_q.legend(fontsize=7)
ax_q.axhline(np.mean(v3_scores), color='#1565C0', ls=':', lw=1.5, alpha=0.8)
ax_q.axhline(np.mean(v4_scores), color='#2E7D32', ls=':', lw=1.5, alpha=0.8)
ax_q.text(5.5, np.mean(v3_scores) + 0.01, f'avg v3={np.mean(v3_scores):.3f}',
           ha='right', fontsize=7, color='#1565C0')
ax_q.text(5.5, np.mean(v4_scores) + 0.01, f'avg v4={np.mean(v4_scores):.3f}',
           ha='right', fontsize=7, color='#2E7D32')

# -- Train HI: Tred by Regime --
ax_regime = fig.add_subplot(gs[1, 2])
regime_map   = {'Regime1(Low)': '#1565C0', 'Regime2(High)': '#C62828'}
regime_label = {'레짐1(저속)': 'Regime1(Low)', '레짐2(고속)': 'Regime2(High)'}
regimes      = hi_summary_train['regime'].unique()
bearings_lst = hi_summary_train['bearing'].unique()
bar_width    = 0.2
x_br         = np.arange(len(bearings_lst))

for ri, regime in enumerate(regimes):
    subset   = hi_summary_train[hi_summary_train['regime'] == regime]
    tred_vals = subset['tred'].values
    eng_label = regime_label.get(regime, regime)
    ax_regime.bar(x_br + ri * bar_width, tred_vals, bar_width,
                   label=eng_label, color=list(regime_map.values())[ri],
                   alpha=0.8, edgecolor='black', lw=0.5)

ax_regime.set_title('Train HI: Tred Score by Regime', fontsize=11, fontweight='bold')
ax_regime.set_xticks(x_br + bar_width / 2)
ax_regime.set_xticklabels([f'B{b}' for b in bearings_lst], fontsize=10)
ax_regime.set_ylabel('Tred Score', fontsize=9)
ax_regime.set_ylim(0.8, 1.02)
ax_regime.legend(fontsize=8)

# -- Test HI curves --
ax_test_hi = fig.add_subplot(gs[2, :2])
for i in range(1, 7):
    hi = test_hi[i]
    q  = hi_summary_test[hi_summary_test['test_id'] == i]['q_score'].values[0]
    q  = hi_summary_test[hi_summary_test['test_id'] == i]['q_score'].values[0]
    ax_test_hi.plot(np.arange(len(hi)), hi, color=TEST_COLORS[i-1], lw=2.0,
                    label=f'Test{i} (Q={q:.3f}, start={hi[0]:.3f})')

ax_test_hi.axhline(0.8, color='red', ls='--', lw=1.2, alpha=0.6, label='Threshold 0.8')
ax_test_hi.set_title('Test HI Curves v4 — Absolute Level (all 6 bearings)', fontsize=12, fontweight='bold')
ax_test_hi.set_xlabel('Slot Index (0-49)', fontsize=10)
ax_test_hi.set_ylabel('Health Index (absolute)', fontsize=10)
ax_test_hi.set_ylim(-0.05, 1.1)
ax_test_hi.legend(fontsize=9, loc='upper left', ncol=2)
ax_test_hi.annotate('Test5/6: start HI elevated\n(SP: late degradation)',
                     xy=(0, test_hi[5][0]), xytext=(8, 0.65),
                     arrowprops=dict(arrowstyle='->', color='#607D8B', lw=1.5),
                     fontsize=9, color='#607D8B', fontweight='bold')
ax_test_hi.annotate('Test2: flat HI\n(no degradation trend)',
                     xy=(25, test_hi[2][25]), xytext=(28, 0.30),
                     arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.5),
                     fontsize=9, color='#9C27B0', fontweight='bold')

# -- LOOCV score history --
ax_loocv = fig.add_subplot(gs[2, 2])
loocv_history = [
    {'run': 'Run1\n(MSE)',      'b1': 0.4359, 'b2': 0.4353, 'b3': 0.3612, 'b4': 0.5567, 'total': 0.4579},
    {'run': 'Run2\n(Asym)',     'b1': 0.7146, 'b2': 0.5777, 'b3': 0.5993, 'b4': 0.4297, 'total': 0.5749},
    {'run': 'Run3',             'b1': 0.4360, 'b2': 0.5385, 'b3': 0.5366, 'b4': 0.4304, 'total': 0.4780},
    {'run': 'Run4',             'b1': 0.4361, 'b2': 0.4389, 'b3': 0.7145, 'b4': 0.4311, 'total': 0.4869},
    {'run': 'Run5',             'b1': 0.2500, 'b2': 0.2500, 'b3': 0.7160, 'b4': 0.5675, 'total': 0.4311},
    {'run': 'Run6\n(LOO+Norm)', 'b1': 0.6950, 'b2': 0.5210, 'b3': 0.6190, 'b4': 0.5512, 'total': 0.5956},
]
lh_df  = pd.DataFrame(loocv_history)
x_run  = np.arange(len(lh_df))

for i, (b, c) in enumerate(zip(['b1', 'b2', 'b3', 'b4'], TRAIN_COLORS)):
    ax_loocv.plot(x_run, lh_df[b], marker='o', color=c, lw=1.5,
                   alpha=0.7, label=f'Bearing{i+1}', markersize=5)
ax_loocv.plot(x_run, lh_df['total'], marker='D', color='black', lw=2.5,
               label='Overall avg', markersize=7, zorder=5)

best_idx = lh_df['total'].idxmax()
ax_loocv.scatter([best_idx], [lh_df['total'].iloc[best_idx]], s=120, color='gold',
                  zorder=6, edgecolors='black', lw=1.5)
ax_loocv.annotate(f'Best\n{lh_df["total"].iloc[best_idx]:.4f}',
                   (best_idx, lh_df['total'].iloc[best_idx]),
                   xytext=(best_idx - 1.2, lh_df['total'].iloc[best_idx] + 0.05),
                   fontsize=9, color='black', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

ax_loocv.set_title('LOOCV RUL Score History', fontsize=11, fontweight='bold')
ax_loocv.set_xticks(x_run)
ax_loocv.set_xticklabels(lh_df['run'], fontsize=8)
ax_loocv.set_ylabel('Score', fontsize=9)
ax_loocv.set_ylim(0.15, 0.85)
ax_loocv.legend(fontsize=8, loc='upper left')

# -- RUL prediction timeline --
ax_rul_tl = fig.add_subplot(gs[3, :])
obs_hours  = 8.17

for i, row in rul_summary.iterrows():
    tid      = int(row['test_id'])
    y        = tid
    rul_hrs  = row['final_rul_hours']
    pred_fail = row['pred_failure_hours']
    q        = hi_summary_test[hi_summary_test['test_id'] == tid]['q_score'].values[0]
    hi_end   = test_hi[tid][-1]

    ax_rul_tl.barh(y, obs_hours, height=0.5, left=0,
                    color=TEST_COLORS[i], alpha=0.6)
    ax_rul_tl.barh(y, rul_hrs, height=0.5, left=obs_hours,
                    color=TEST_COLORS[i], alpha=0.25, hatch='//')
    ax_rul_tl.axvline(pred_fail, color=TEST_COLORS[i], ls=':', lw=1, alpha=0.7)

    txt_color = 'white' if hi_end > 0.5 else 'black'
    ax_rul_tl.text(obs_hours / 2, y + 0.28,
                    f'Test{tid}  |  HI_end={hi_end:.3f}  |  RUL={row["final_rul_cycles"]:.1f}cyc ({rul_hrs:.2f}hr)  |  Q={q:.3f}',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=txt_color)
    ax_rul_tl.text(obs_hours / 2, y - 0.28, 'Observed (8.17 hr)',
                    ha='center', va='top', fontsize=7.5, color=txt_color)
    if rul_hrs > 0.3:
        ax_rul_tl.text(obs_hours + rul_hrs / 2, y, f'+{rul_hrs:.2f}hr',
                        ha='center', va='center', fontsize=8)

ax_rul_tl.axvline(obs_hours, color='black', ls='--', lw=2)
ax_rul_tl.set_xlabel('Time (hours)', fontsize=11)
ax_rul_tl.set_title('Test RUL Prediction Timeline (observed + predicted remaining life)', fontsize=12, fontweight='bold')
ax_rul_tl.set_yticks(range(1, 7))
ax_rul_tl.set_yticklabels([f'Test{i}' for i in range(1, 7)], fontsize=10)
ax_rul_tl.set_xlim(0, 22)

from matplotlib.patches import Patch
ax_rul_tl.legend(handles=[
    Patch(facecolor='gray', alpha=0.6, label='Observed window (8.17 hr)'),
    Patch(facecolor='gray', alpha=0.25, hatch='//', label='Predicted RUL'),
    plt.Line2D([0], [0], color='black', ls='--', lw=2, label='Current time'),
], loc='upper right', fontsize=9)

plt.savefig(f'{BASE}/analysis_summary.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print(f'Saved: {BASE}/analysis_summary.png')
plt.close()

# ===============================================================
# Figure 2: Per-Test RUL time series
# ===============================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('LSTM RUL Prediction Time Series — Per Test Bearing', fontsize=16, fontweight='bold')
fig2.patch.set_facecolor('#F8F9FA')

for idx, (ax, tid) in enumerate(zip(axes2.flat, range(1, 7))):
    sub    = rul_all[rul_all['test_id'] == tid].copy()
    hi     = test_hi[tid]
    hi_end = hi[-1]
    q      = hi_summary_test[hi_summary_test['test_id'] == tid]['q_score'].values[0]

    ax2 = ax.twinx()
    ax2.plot(sub['obs_cycle'], hi[:len(sub)], color='#BDBDBD', lw=2, alpha=0.8, label='HI')
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_ylabel('Health Index', fontsize=8, color='gray')
    ax2.tick_params(axis='y', colors='gray', labelsize=7)

    ax.plot(sub['obs_cycle'], sub['rul_pred_cycles'], color=TEST_COLORS[idx],
             lw=2, marker='o', markersize=3, label='Pred RUL (cycles)')

    z = np.polyfit(sub['obs_cycle'], sub['rul_pred_cycles'], 1)
    p = np.poly1d(z)
    ax.plot(sub['obs_cycle'], p(sub['obs_cycle']), '--', color='black',
             alpha=0.5, lw=1, label=f'Trend (slope={z[0]:.1f})')

    final_rul = rul_summary[rul_summary['test_id'] == tid]['final_rul_cycles'].values[0]
    ax.scatter([sub['obs_cycle'].iloc[-1]], [sub['rul_pred_cycles'].iloc[-1]],
                color='red', s=80, zorder=5)
    ax.annotate(f'Final: {final_rul:.1f}cyc',
                 (sub['obs_cycle'].iloc[-1], final_rul),
                 xytext=(-30, 10), textcoords='offset points',
                 fontsize=8, color='red', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.set_title(f'Test{tid}  (HI_end={hi_end:.3f}, Q={q:.3f})',
                  fontsize=10, fontweight='bold')
    ax.set_xlabel('Observed Cycle', fontsize=8)
    ax.set_ylabel('Pred RUL (cycles)', fontsize=8, color=TEST_COLORS[idx])
    ax.tick_params(axis='y', colors=TEST_COLORS[idx], labelsize=8)

    if z[0] > 0:
        ax.text(0.5, 0.9, '[!] RUL increasing (unstable)',
                 transform=ax.transAxes, ha='center', fontsize=8,
                 color='orange', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig(f'{BASE}/analysis_rul_detail.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print(f'Saved: {BASE}/analysis_rul_detail.png')
plt.close()

# ===============================================================
# Figure 3: Improvement summary heatmap + bar chart
# ===============================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle('0514 Pipeline Improvement Summary', fontsize=16, fontweight='bold')
fig3.patch.set_facecolor('#F8F9FA')

# Heatmap
ax_hm = axes3[0]
data  = np.array([v1_scores, v3_scores, v4_scores,
                  [v3 - v1 for v1, v3 in zip(v1_scores, v3_scores)],
                  [v4 - v3 for v3, v4 in zip(v3_scores, v4_scores)]])
im    = ax_hm.imshow(data, aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=0.5)
ax_hm.set_xticks(range(6))
ax_hm.set_xticklabels([f'Test{i}' for i in range(1, 7)], fontsize=10)
ax_hm.set_yticks([0, 1, 2, 3, 4])
ax_hm.set_yticklabels(['v1 Q', 'v3 Q', 'v4 Q', 'Δ v3-v1', 'Δ v4-v3'], fontsize=9)
for r in range(5):
    for c in range(6):
        val   = data[r, c]
        txt   = f'{val:+.3f}' if r >= 3 else f'{val:.3f}'
        color = 'white' if abs(val) > 0.3 else 'black'
        ax_hm.text(c, r, txt, ha='center', va='center', fontsize=10,
                    fontweight='bold', color=color)
plt.colorbar(im, ax=ax_hm, shrink=0.8)
ax_hm.set_title('HI Q-Score Heatmap (v1 / v3 / v4)', fontsize=12, fontweight='bold')

# Bar chart: LOOCV initial vs final
ax_bar = axes3[1]
bearing_labels = ['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4', 'Overall']
initial = [0.4359, 0.4353, 0.3612, 0.5567, 0.4579]
final   = [0.6950, 0.5210, 0.6190, 0.5512, 0.5956]
x_b     = np.arange(5)

bars_init = ax_bar.bar(x_b - 0.2, initial, 0.4, label='Initial (MSELoss)',
                        color='#B0BEC5', alpha=0.9, edgecolor='gray')
bars_fin  = ax_bar.bar(x_b + 0.2, final, 0.4, label='Final (AsymLoss + LOO + RUL norm)',
                        color='#42A5F5', alpha=0.9, edgecolor='#1565C0')

for b_init, b_fin, x in zip(bars_init, bars_fin, x_b):
    delta = b_fin.get_height() - b_init.get_height()
    sign  = '+' if delta >= 0 else ''
    color = '#1B5E20' if delta > 0 else '#B71C1C'
    ax_bar.text(x, max(b_init.get_height(), b_fin.get_height()) + 0.01,
                 f'{sign}{delta:.3f}', ha='center', fontsize=9,
                 color=color, fontweight='bold')

ax_bar.set_title('LOOCV RUL Score: Initial vs Final', fontsize=12, fontweight='bold')
ax_bar.set_xticks(x_b)
ax_bar.set_xticklabels(bearing_labels, fontsize=9)
ax_bar.set_ylabel('Score', fontsize=10)
ax_bar.set_ylim(0, 0.9)
ax_bar.legend(fontsize=9)
ax_bar.axhline(0.4579, color='gray', ls=':', lw=1.5, alpha=0.7)
ax_bar.axhline(0.5956, color='#1565C0', ls=':', lw=1.5, alpha=0.7)
ax_bar.text(4.4, 0.4579 + 0.01, '0.4579', ha='right', fontsize=8, color='gray')
ax_bar.text(4.4, 0.5956 + 0.01, '0.5956', ha='right', fontsize=8, color='#1565C0')

plt.tight_layout()
plt.savefig(f'{BASE}/analysis_improvement.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print(f'Saved: {BASE}/analysis_improvement.png')
plt.close()

print('\nAll done.')
