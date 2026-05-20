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
fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.97, top=0.95, bottom=0.04,
                       height_ratios=[0.8, 1, 1])
fig.suptitle('0514 Experiment Results — Summary Dashboard  [HI v4: Train-Anchored]', fontsize=18, fontweight='bold', y=0.98)

obs_hours = 8.17

# -- RUL bar chart (cycles), full width --
ax_rul = fig.add_subplot(gs[0, :])
rul_cyc = [rul_summary[rul_summary['test_id'] == t]['final_rul_cycles'].values[0] for t in range(1, 7)]
rul_hrs = [rul_summary[rul_summary['test_id'] == t]['final_rul_hours'].values[0]  for t in range(1, 7)]
y_pos   = np.arange(6, 0, -1)  # Test1 at top

ax_rul.barh(y_pos, rul_cyc, height=0.55, color=TEST_COLORS, alpha=0.85, edgecolor='black', lw=0.5)
for y, cyc, hrs in zip(y_pos, rul_cyc, rul_hrs):
    ax_rul.text(cyc + 0.3, y, f'{cyc:.1f} cycles  ({hrs:.2f} hr)',
                va='center', fontsize=10, fontweight='bold')
ax_rul.set_yticks(y_pos)
ax_rul.set_yticklabels([f'Test{t}' for t in range(1, 7)], fontsize=11)
ax_rul.set_xlabel('Predicted RUL (cycles)', fontsize=10)
ax_rul.set_title('Predicted RUL per Test Bearing', fontsize=12, fontweight='bold')
ax_rul.set_xlim(0, max(rul_cyc) * 1.2)

# -- Train HI curves (Bearing 1-4) --
loocv_scores = {1: 0.695, 2: 0.521, 3: 0.619, 4: 0.551}
train_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
              fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

for idx, (ax, b) in enumerate(zip(train_axes, range(1, 5))):
    hi = train_hi[b]
    x  = np.arange(len(hi))
    ax.plot(x, hi, color=TRAIN_COLORS[idx], lw=2, label=f'Bearing{b} HI')
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
