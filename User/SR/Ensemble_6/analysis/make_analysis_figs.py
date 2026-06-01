"""Generate figures for analysis.md (2026-06-01)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUT = "."
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})

# ── Data ────────────────────────────────────────────────────────────────────
tests      = ["T1", "T2", "T3", "T4", "T5", "T6"]
start_obs  = [16,   42,   22,   25,   79,   99]
hi_start   = [0.000,0.498,0.417,0.000,0.665,0.841]
hi_end     = [0.386,0.417,0.596,0.697,0.835,0.869]
rul_base   = [2.07, 11.38, 7.99, 4.16, 2.01, 1.53]
rul_expl   = [4.62, 13.85, 9.34, 3.63, 4.51, 3.43]

# LOOCV per-bearing
bearings        = ["B1",   "B2",   "B3",   "B4"]
dtw_lf          = [0.4823, 0.5300, 0.6493, 0.4198]
expl_lf         = [0.6135, 0.5965, 0.6172, 0.5564]
baseline_leaked = [0.5808, 0.5344, 0.6617, 0.4226]
mean_er_base    = [-26.5,  +14.5,  -23.0,  +48.9]   # % (+ = under, - = over)
mean_er_expl    = [-16.6,   -7.7,  -20.6,   +3.9]

# Experiment timeline
exp_names = ["Baseline\n(leaked)", "DTW\nLeak-free", "Exp-I\n(leaked*)", "Exp-J\n(leaked*)",
             "Exp-K\n(leaked*)", "Exp-L\nAsym (LF)"]
exp_scores = [0.5499, 0.5204, 0.5826, 0.6064, 0.6080, 0.5959]
exp_leaked = [True,   False,  True,   True,   True,   False]

# ── Figure 1: Test RUL comparison ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 1 — Test RUL: Baseline vs Exp-L_Asym", fontweight="bold")

ax = axes[0]
x = np.arange(len(tests))
w = 0.35
bars1 = ax.bar(x - w/2, rul_base, w, label="Baseline (V1b_dtw)", color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + w/2, rul_expl, w, label="Exp-L_Asym",          color="#DD8452", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(tests)
ax.set_ylabel("RUL (hours)")
ax.set_title("Predicted RUL by Test Bearing")
ax.legend()
ax.axhline(0, color="k", lw=0.5)
for b, r1, r2 in zip(x, rul_base, rul_expl):
    ratio = r2 / r1
    color = "#c0392b" if ratio > 1.5 else ("#e67e22" if ratio > 1.2 else "#27ae60")
    ax.text(b, max(r1, r2) + 0.3, f"×{ratio:.1f}", ha="center", fontsize=9, color=color, fontweight="bold")

ax = axes[1]
colors = []
for s, e in zip(start_obs, hi_end):
    if s >= 70:
        colors.append("#c0392b")
    elif hi_end[tests.index("T" + str(tests.index("T" + str(tests.index("T1")+1)) if False else 1))] > 0.7:
        colors.append("#e67e22")
    else:
        colors.append("#4C72B0")
colors = []
for s, he in zip(start_obs, hi_end):
    if s >= 70:
        colors.append("#c0392b")
    else:
        colors.append("#4C72B0")

sc = ax.scatter(start_obs, hi_end, c=colors, s=120, zorder=5)
for i, (t, s, he, rb, re) in enumerate(zip(tests, start_obs, hi_end, rul_base, rul_expl)):
    ax.annotate(f"{t}\nbase={rb:.1f}h\nnew={re:.1f}h",
                (s, he), xytext=(6, 4), textcoords="offset points", fontsize=8)
ax.axhline(0.75, color="gray", lw=1, ls="--", label="Mean EOL HI = 0.75")
ax.axvline(116.5 * 0.65, color="#c0392b", lw=1, ls=":", alpha=0.6, label="65% of mean life")
ax.set_xlabel("start_obs (estimated life position)")
ax.set_ylabel("hi_end (last observed HI)")
ax.set_title("Test Bearing Position Map")
ax.legend(fontsize=9)
red_patch   = mpatches.Patch(color="#c0392b", label="start_obs ≥ 70 (near EOL)")
blue_patch  = mpatches.Patch(color="#4C72B0", label="start_obs < 70")
ax.legend(handles=[red_patch, blue_patch], fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_test_rul_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig1 done")

# ── Figure 2: Why longer? Three mechanisms ──────────────────────────────────
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Fig 2 — Why Exp-L_Asym Predicts Longer RUL Than Baseline", fontweight="bold", fontsize=13)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel A: obs_frac comparison for test bearings
ax_a = fig.add_subplot(gs[0, 0])
cycle_obs_frac = [(16 + 0) / 116.5, (42 + 0) / 116.5,
                  (22 + 0) / 116.5, (25 + 0) / 116.5,
                  (79 + 0) / 116.5, (99 + 0) / 116.5]   # at start of test window (proxy)
# use hi_end-based obs_frac at last observed point
hi_obs_frac = [min(he / 0.75, 1.2) for he in hi_end]

x = np.arange(len(tests))
w = 0.35
ax_a.bar(x - w/2, cycle_obs_frac, w, label="Cycle-based (start_obs/116.5)", color="#4C72B0", alpha=0.8)
ax_a.bar(x + w/2, hi_obs_frac,    w, label="HI-based (hi_end/0.75)",        color="#DD8452", alpha=0.8)
ax_a.axhline(1.0, color="red", lw=1, ls="--", alpha=0.6, label="obs_frac = 1.0 (EOL)")
ax_a.set_xticks(x); ax_a.set_xticklabels(tests, fontsize=9)
ax_a.set_ylabel("obs_frac at last point")
ax_a.set_title("A. obs_frac Input to LGBM/NN")
ax_a.legend(fontsize=7.5)
ax_a.set_ylim(0, 1.35)

# Panel B: capped upside mechanism illustration
ax_b = fig.add_subplot(gs[0, 1])
dtw_ex  = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
nn_preds = np.array([1.0, 1.5, 2.0, 3.0, 4.5])
cap_val  = 2.0
alpha    = 0.6
upside   = alpha * np.clip(nn_preds - dtw_ex, 0, (cap_val - 1) * dtw_ex)
final    = dtw_ex + upside

ax_b.plot(nn_preds, dtw_ex, "b--", lw=1.5, label="DTW base = 2hr (constant)")
ax_b.plot(nn_preds, nn_preds, "g:",  lw=1.5, label="NN pred (45° line)")
ax_b.plot(nn_preds, final,   "r-",  lw=2,   label=f"Final (α={alpha}, cap={cap_val})")
ax_b.fill_between(nn_preds, dtw_ex, final, alpha=0.15, color="red", label="Upside added")
ax_b.set_xlabel("NN prediction (hr)")
ax_b.set_ylabel("Final prediction (hr)")
ax_b.set_title("B. Capped Upside Mechanism")
ax_b.legend(fontsize=7.5)
ax_b.set_xlim(0.5, 5)
ax_b.annotate("max +100%\n(cap=2.0)", xy=(4.5, 4.0), xytext=(3.0, 3.8),
              arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="red")

# Panel C: B4 systematic bias correction
ax_c = fig.add_subplot(gs[0, 2])
ax_c.bar(bearings, mean_er_base, color=["#e74c3c" if e > 20 else "#f39c12" if e > 0 else "#2ecc71"
                                          for e in mean_er_base],
         alpha=0.8, label="Baseline mean_er %")
ax_c.bar([b + " " for b in bearings], mean_er_expl,
         color=["#27ae60" if abs(e) < 15 else "#e67e22" for e in mean_er_expl],
         alpha=0.7, label="Exp-L_Asym mean_er %")
ax_c.axhline(0, color="k", lw=1)
ax_c.axhline(20,  color="gray", lw=0.7, ls="--", alpha=0.5)
ax_c.axhline(-20, color="gray", lw=0.7, ls="--", alpha=0.5)
ax_c.set_ylabel("Mean Error % (+under / -over)")
ax_c.set_title("C. Systematic Bias in Training\n(+: under-predict, –: over-predict)")
ax_c.set_ylim(-65, 65)
for i, (b, e) in enumerate(zip(bearings, mean_er_base)):
    ax_c.text(i - 0.2, e + (3 if e >= 0 else -6), f"{e:+.0f}%", ha="center", fontsize=8, color="k")
for i, (b, e) in enumerate(zip(bearings, mean_er_expl)):
    ax_c.text(i + 0.2, e + (3 if e >= 0 else -6), f"{e:+.0f}%", ha="center", fontsize=8, color="gray")
base_patch = mpatches.Patch(color="#e74c3c", alpha=0.8, label="Baseline (left bars)")
expl_patch = mpatches.Patch(color="#27ae60", alpha=0.7, label="Exp-L_Asym (right bars)")
ax_c.legend(handles=[base_patch, expl_patch], fontsize=8)

# Panel D: Per-bearing LOOCV comparison
ax_d = fig.add_subplot(gs[1, 0])
x = np.arange(len(bearings))
w = 0.28
ax_d.bar(x - w,   baseline_leaked, w, label="Baseline (leaked)",     color="#95a5a6", alpha=0.8)
ax_d.bar(x,       dtw_lf,          w, label="DTW Leak-free",         color="#4C72B0", alpha=0.85)
ax_d.bar(x + w,   expl_lf,         w, label="Exp-L_Asym (LF)",       color="#DD8452", alpha=0.85)
ax_d.set_xticks(x); ax_d.set_xticklabels(bearings)
ax_d.set_ylabel("LOOCV Score")
ax_d.set_title("D. Per-bearing LOOCV Scores\n(LF = Leak-Free)")
ax_d.legend(fontsize=8)
ax_d.set_ylim(0.3, 0.75)
ax_d.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)

# Panel E: T5/T6 physical suspicion
ax_e = fig.add_subplot(gs[1, 1])
t56_labels = ["T5\nstart=79\nhi_e=0.835", "T6\nstart=99\nhi_e=0.869"]
t56_base   = [2.01, 1.53]
t56_expl   = [4.51, 3.43]
x56 = np.arange(2)
ax_e.bar(x56 - 0.2, t56_base, 0.35, color="#4C72B0", alpha=0.85, label="Baseline")
ax_e.bar(x56 + 0.2, t56_expl, 0.35, color="#DD8452", alpha=0.85, label="Exp-L_Asym")
ax_e.set_xticks(x56); ax_e.set_xticklabels(t56_labels, fontsize=9)
ax_e.set_ylabel("Predicted RUL (hr)")
ax_e.set_title("E. T5/T6 — Near-EOL Bearings\n(Suspicious Doubling)")
ax_e.legend(fontsize=9)
for i, (rb, re) in enumerate(zip(t56_base, t56_expl)):
    ax_e.text(i, max(rb, re) + 0.1, f"×{re/rb:.1f}", ha="center", fontsize=11,
              color="#c0392b", fontweight="bold")
ax_e.annotate("HI already > 0.75\n(past mean EOL HI)\n→ Should be near EOL",
              xy=(0.5, 3.6), fontsize=8.5, color="#c0392b",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", alpha=0.8))

# Panel F: Comparison of T5/T6 with B4 pattern
ax_f = fig.add_subplot(gs[1, 2])
categories = ["B4 train\n(hi_start≈0.30)", "T5\n(hi_start=0.665)", "T6\n(hi_start=0.841)"]
hi_starts_comparison = [0.30, 0.665, 0.841]
under_pred_pct = [49.0, None, None]   # baseline under-predict %
note_colors = ["#e74c3c", "#e67e22", "#c0392b"]

ax_f.barh(categories, hi_starts_comparison, color=note_colors, alpha=0.75)
ax_f.axvline(0.75, color="gray", lw=1.5, ls="--", label="Mean EOL HI = 0.75")
ax_f.set_xlabel("hi_start (initial observed HI)")
ax_f.set_title("F. T5/T6 vs B4 Pattern\n(High initial HI → B4-like?)")
ax_f.legend(fontsize=9)
ax_f.text(0.31, 0, " B4: baseline under-pred +49%\n → new pipeline corrects to +4%",
          va="center", fontsize=7.5, color="#e74c3c")
ax_f.text(0.68, 1, " T5: if B4-like,\n longer RUL plausible",
          va="center", fontsize=7.5, color="#e67e22")
ax_f.text(0.85, 2, " T6: very high HI\n → ambiguous",
          va="center", fontsize=7.5, color="#c0392b")

plt.savefig(f"{OUT}/fig2_why_longer.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig2 done")

# ── Figure 3: Score progression + overfitting risk ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 3 — Score Progression & Overfitting Risk", fontweight="bold")

ax = axes[0]
x = np.arange(len(exp_names))
bar_colors = ["#95a5a6" if lk else "#2ecc71" for lk in exp_leaked]
bars = ax.bar(x, exp_scores, color=bar_colors, alpha=0.85, width=0.6)
ax.set_xticks(x); ax.set_xticklabels(exp_names, fontsize=8.5)
ax.set_ylabel("Overall LOOCV Score")
ax.set_title("Experiment Score Progression\n(green = leak-free, gray = leaked)")
ax.set_ylim(0.45, 0.65)
for xi, (sc_val, lk) in enumerate(zip(exp_scores, exp_leaked)):
    ax.text(xi, sc_val + 0.003, f"{sc_val:.4f}", ha="center", fontsize=8.5,
            fontweight="bold", color="k")
    if lk:
        ax.text(xi, sc_val - 0.015, "⚠leaked", ha="center", fontsize=7, color="#c0392b")
lf_patch   = mpatches.Patch(color="#2ecc71", alpha=0.85, label="Leak-free (valid)")
lk_patch   = mpatches.Patch(color="#95a5a6", alpha=0.85, label="Leaked (inflated)")
ax.legend(handles=[lf_patch, lk_patch], fontsize=9)
ax.axhline(0.5204, color="#4C72B0", lw=1.5, ls="--", alpha=0.7, label="DTW LF baseline")

ax = axes[1]
# Overfitting risk assessment
models_info = [
    ("Baseline\n(DTW only)",       1,   0.5204, False),
    ("Exp-I\n(beta=0.0)",          4,   0.55,   True),    # approx LF
    ("Exp-J\n(DTW+capped)",        6,   0.576,  True),    # approx LF
    ("Exp-L_Asym\n(full)",         12,  0.5959, False),
]
names_risk = [m[0] for m in models_info]
n_params   = [m[1] for m in models_info]
scores_r   = [m[2] for m in models_info]
approx     = [m[3] for m in models_info]

colors_r = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
for i, (nm, np_val, sc_val, apx) in enumerate(zip(names_risk, n_params, scores_r, approx)):
    marker = "s" if apx else "o"
    ax.scatter(np_val, sc_val, s=180, color=colors_r[i], zorder=5, marker=marker)
    ax.annotate(nm, (np_val, sc_val), xytext=(5, 5), textcoords="offset points", fontsize=8.5)

ax.plot(n_params, scores_r, "k--", lw=1, alpha=0.4)
ax.set_xlabel("Effective Hyperparameter Count")
ax.set_ylabel("Estimated Leak-free LOOCV Score")
ax.set_title("Overfitting Risk vs Performance\n(○ = measured, □ = estimated)")
ax.set_xlim(0, 14)
ax.set_ylim(0.5, 0.62)
ax.axvspan(8, 14, alpha=0.07, color="red", label="High overfitting risk zone")
ax.axvspan(0,  8, alpha=0.05, color="green", label="Low overfitting risk zone")
ax.legend(fontsize=8)

for i, (np_val, sc_val, col) in enumerate(zip(n_params, scores_r, colors_r)):
    ax.annotate("", xy=(np_val, sc_val),
                xytext=(np_val - 0.1, sc_val - 0.001))

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_score_progression.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig3 done")

# ── Figure 4: Decision matrix ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Fig 4 — Submission Decision Matrix", fontweight="bold")

decisions = [
    ("Baseline\n(V1b_dtw)",  0.5204, "Low",    "No change\n(6th place)"),
    ("Exp-L_Asym",           0.5959, "High",   "Best LOOCV but\nT5/T6 suspicious"),
    ("Exp-J approx",         0.576,  "Medium", "Structural improvement\nfewer params"),
]

risk_map   = {"Low": 1, "Medium": 2, "High": 3}
risk_color = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

for nm, sc, risk, note in decisions:
    rx = risk_map[risk]
    ax.scatter(rx, sc, s=300, color=risk_color[risk], zorder=5)
    ax.annotate(f"{nm}\nScore={sc:.4f}\n{note}",
                (rx, sc), xytext=(15, 0), textcoords="offset points",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=risk_color[risk]))

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Low Risk", "Medium Risk", "High Risk"])
ax.set_ylabel("Estimated Leak-free LOOCV Score")
ax.set_title("Risk vs Performance Trade-off for Submission")
ax.set_xlim(0.3, 4.5)
ax.set_ylim(0.48, 0.62)
ax.axhspan(0.58, 0.62, alpha=0.06, color="gold", label="Target zone (>0.58)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_decision_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig4 done")

print("All figures saved.")
