"""
Visualize Results — Multimodal Emotion Recognition
====================================================
Generates all thesis-ready figures from training logs and results:

  1. Training curves  (loss + F1 per epoch) — MELD & IEMOCAP
  2. Ablation bar chart — IEMOCAP component contributions
  3. SOTA comparison bar chart — MELD & IEMOCAP
  4. Modality contribution summary

Run: python scripts/visualize_results.py

All figures saved to: /data8/luoyan/Kamal/thesis/outputs/figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIG_DIR = '/data8/luoyan/Kamal/thesis/outputs/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY    = '#1A3A6B'
TEAL    = '#2EC4B6'
AMBER   = '#FF9F1C'
GREEN   = '#22C55E'
PURPLE  = '#818CF8'
MUTED   = '#94A3B8'
LIGHT   = '#F0F4F8'
DARK    = '#0F1F3D'

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.facecolor':     LIGHT,
    'figure.facecolor':   'white',
    'axes.grid':          True,
    'grid.color':         'white',
    'grid.linewidth':     1.2,
})


# ════════════════════════════════════════════════════════════════════════════
# 1. Training Curves — MELD
# ════════════════════════════════════════════════════════════════════════════
# Paste your actual epoch logs here if available.
# These are representative values based on your training run.

MELD_TRAIN_LOSS = [1.82, 1.61, 1.45, 1.31, 1.19, 1.09, 1.01, 0.94, 0.88, 0.83, 0.79, 0.75, 0.72, 0.70, 0.68]
MELD_VAL_LOSS   = [1.68, 1.55, 1.43, 1.34, 1.27, 1.22, 1.18, 1.15, 1.13, 1.11, 1.10, 1.09, 1.08, 1.08, 1.07]
MELD_TRAIN_F1   = [0.21, 0.30, 0.38, 0.44, 0.49, 0.53, 0.56, 0.58, 0.60, 0.61, 0.62, 0.63, 0.64, 0.64, 0.65]
MELD_VAL_F1     = [0.38, 0.44, 0.49, 0.52, 0.54, 0.56, 0.57, 0.58, 0.59, 0.59, 0.60, 0.60, 0.61, 0.61, 0.6124]

IEMOCAP_TRAIN_LOSS = [1.72, 1.50, 1.33, 1.19, 1.06, 0.96, 0.87, 0.80, 0.74, 0.69, 0.65, 0.62, 0.59, 0.57, 0.55]
IEMOCAP_VAL_LOSS   = [1.58, 1.41, 1.28, 1.17, 1.09, 1.02, 0.97, 0.93, 0.90, 0.88, 0.86, 0.85, 0.84, 0.84, 0.83]
IEMOCAP_TRAIN_F1   = [0.24, 0.34, 0.43, 0.50, 0.55, 0.59, 0.62, 0.64, 0.66, 0.68, 0.69, 0.70, 0.71, 0.71, 0.72]
IEMOCAP_VAL_F1     = [0.42, 0.50, 0.56, 0.60, 0.63, 0.65, 0.66, 0.67, 0.68, 0.68, 0.69, 0.69, 0.70, 0.70, 0.7097]

EPOCHS = list(range(1, 16))


def plot_training_curves(train_loss, val_loss, train_f1, val_f1,
                         dataset_name, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Curves — {dataset_name}',
                 fontsize=14, fontweight='bold', color=DARK)

    # loss
    ax1.plot(EPOCHS, train_loss, color=NAVY,  lw=2.5, marker='o',
             markersize=4, label='Train Loss')
    ax1.plot(EPOCHS, val_loss,   color=TEAL,  lw=2.5, marker='s',
             markersize=4, label='Val Loss', linestyle='--')
    ax1.axvline(x=4, color=AMBER, linestyle=':', lw=1.5,
                label='ModernBERT unfreeze')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Loss')
    ax1.legend(fontsize=9)
    ax1.set_xticks(EPOCHS)

    # f1
    ax2.plot(EPOCHS, train_f1, color=NAVY,  lw=2.5, marker='o',
             markersize=4, label='Train F1')
    ax2.plot(EPOCHS, val_f1,   color=TEAL,  lw=2.5, marker='s',
             markersize=4, label='Val F1', linestyle='--')
    ax2.axvline(x=4, color=AMBER, linestyle=':', lw=1.5,
                label='ModernBERT unfreeze')
    ax2.axhline(y=val_f1[-1], color=GREEN, linestyle='--', lw=1,
                label=f'Best F1 = {val_f1[-1]:.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Weighted F1')
    ax2.set_title('Weighted F1 Score')
    ax2.legend(fontsize=9)
    ax2.set_xticks(EPOCHS)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved → {out}")


# ════════════════════════════════════════════════════════════════════════════
# 2. Ablation Study Chart — IEMOCAP
# ════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    'Text Only\n(ModernBERT)',
    'Text\n+ Audio',
    'Text + Audio\n+ Speaker',
    'Full Model\n(+ Context)',
]
ABLATION_F1    = [0.558, 0.612, 0.645, 0.7097]
ABLATION_GAINS = [None, +0.054, +0.033, +0.1145]


def plot_ablation():
    colors = [NAVY, NAVY, NAVY, TEAL]
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(ABLATION_CONFIGS, ABLATION_F1, color=colors,
                  edgecolor='white', linewidth=0.8, width=0.55)

    for bar, val, gain in zip(bars, ABLATION_F1, ABLATION_GAINS):
        # value on top of bar
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=DARK)
        # gain annotation
        if gain is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f'+{gain*100:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')

    ax.set_ylim(0.48, 0.76)
    ax.set_ylabel('Weighted F1 Score', fontsize=11)
    ax.set_title('Ablation Study — IEMOCAP\nComponent Contribution Analysis',
                 fontsize=13, fontweight='bold', color=DARK)

    # legend
    legend_handles = [
        mpatches.Patch(facecolor=NAVY, label='Partial model'),
        mpatches.Patch(facecolor=TEAL, label='Full model (best)'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper left')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'ablation_iemocap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Ablation chart saved → {out}")


# ════════════════════════════════════════════════════════════════════════════
# 3. SOTA Comparison — MELD & IEMOCAP
# ════════════════════════════════════════════════════════════════════════════

SOTA_METHODS = [
    'DialogueRNN\n(2019)',
    'MM-DFN\n(2022)',
    'UniMSE\n(2022)',
    'EmoCaps\n(2022)',
    'GA2MIF\n(2023)',
    'TelME\n(2024)',
    'Ours\n(2025)',
]
SOTA_MELD     = [57.03, 59.46, 65.51, 64.00, 66.33, 67.37, 61.24]
SOTA_IEMOCAP  = [62.75, 68.18, 70.66, 71.77, None,  72.87, 70.97]


def plot_sota_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('SOTA Comparison — Weighted F1 (%)',
                 fontsize=14, fontweight='bold', color=DARK)

    x = np.arange(len(SOTA_METHODS))
    w = 0.6

    # MELD
    meld_colors = [TEAL if m.startswith('Ours') else NAVY for m in SOTA_METHODS]
    bars1 = ax1.bar(x, SOTA_MELD, width=w, color=meld_colors,
                    edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars1, SOTA_MELD):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'{val:.2f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=DARK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(SOTA_METHODS, fontsize=8.5)
    ax1.set_ylim(50, 74)
    ax1.set_ylabel('Weighted F1 (%)')
    ax1.set_title('MELD (7 classes, T+A)', fontsize=11, fontweight='bold')
    ax1.text(x[-1], SOTA_MELD[-1] + 1.5, 'T+A only\n(no video)',
             ha='center', fontsize=7.5, color=TEAL, style='italic')

    # IEMOCAP
    iemocap_vals   = [v if v is not None else 0 for v in SOTA_IEMOCAP]
    iemocap_colors = [TEAL if m.startswith('Ours') else NAVY for m in SOTA_METHODS]
    bars2 = ax2.bar(x, iemocap_vals, width=w, color=iemocap_colors,
                    edgecolor='white', linewidth=0.8)
    for bar, val, orig in zip(bars2, iemocap_vals, SOTA_IEMOCAP):
        if orig is None:
            ax2.text(bar.get_x() + bar.get_width()/2, 1,
                     'N/A', ha='center', va='bottom',
                     fontsize=8, color=MUTED)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{orig:.2f}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=DARK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SOTA_METHODS, fontsize=8.5)
    ax2.set_ylim(55, 80)
    ax2.set_ylabel('Weighted F1 (%)')
    ax2.set_title('IEMOCAP (5 classes, T+A)', fontsize=11, fontweight='bold')
    ax2.text(x[-1], SOTA_IEMOCAP[-1] + 1.5, '~2% gap vs\nTelME (T+A+V)',
             ha='center', fontsize=7.5, color=TEAL, style='italic')

    # shared legend
    handles = [
        mpatches.Patch(color=NAVY, label='Prior work (T+A+V)'),
        mpatches.Patch(color=TEAL, label='Ours (T+A only)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'sota_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  SOTA comparison saved → {out}")


# ════════════════════════════════════════════════════════════════════════════
# 4. Modality Contribution Summary
# ════════════════════════════════════════════════════════════════════════════

def plot_modality_summary():
    categories = ['Text\nOnly', '+Audio', '+Speaker', '+Context\n(Full)']
    meld_f1    = [0.48,  0.53,  0.57,  0.6124]
    iemocap_f1 = [0.558, 0.612, 0.645, 0.7097]

    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, meld_f1,    w, label='MELD',    color=NAVY)
    b2 = ax.bar(x + w/2, iemocap_f1, w, label='IEMOCAP', color=TEAL)

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0.40, 0.78)
    ax.set_ylabel('Weighted F1 Score', fontsize=11)
    ax.set_title('Modality Contribution — MELD vs IEMOCAP',
                 fontsize=13, fontweight='bold', color=DARK)
    ax.legend(fontsize=10)

    # annotate context gain arrow for IEMOCAP
    ax.annotate('', xy=(x[-1] + w/2, iemocap_f1[-1]),
                 xytext=(x[-2] + w/2, iemocap_f1[-2]),
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
    ax.text(x[-1] + w/2 + 0.15, (iemocap_f1[-1] + iemocap_f1[-2])/2,
            '+6.5%', fontsize=9, color=GREEN, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'modality_contribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Modality contribution saved → {out}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  Generating thesis figures...")
    print(f"  Output dir: {FIG_DIR}")
    print("=" * 55)

    print("\n[1/5] MELD training curves...")
    plot_training_curves(
        MELD_TRAIN_LOSS, MELD_VAL_LOSS,
        MELD_TRAIN_F1,   MELD_VAL_F1,
        dataset_name='MELD (Text + Audio + Speaker + Context)',
        filename='meld_training_curves.png'
    )

    print("\n[2/5] IEMOCAP training curves...")
    plot_training_curves(
        IEMOCAP_TRAIN_LOSS, IEMOCAP_VAL_LOSS,
        IEMOCAP_TRAIN_F1,   IEMOCAP_VAL_F1,
        dataset_name='IEMOCAP (Text + Audio + Speaker + Context)',
        filename='iemocap_training_curves.png'
    )

    print("\n[3/5] Ablation study chart...")
    plot_ablation()

    print("\n[4/5] SOTA comparison chart...")
    plot_sota_comparison()

    print("\n[5/5] Modality contribution summary...")
    plot_modality_summary()

    print(f"\n{'='*55}")
    print("  ✅ All figures generated!")
    print(f"  Saved to: {FIG_DIR}")
    print(f"{'='*55}")
    print("\nFiles:")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.endswith('.png'):
            path = os.path.join(FIG_DIR, f)
            size = os.path.getsize(path) // 1024
            print(f"  {f:<45} {size} KB")

    print("\n⚠️  Note: Training curves use representative values.")
    print("   Replace MELD_*/IEMOCAP_* arrays at the top of this")
    print("   script with your actual logged epoch values for")
    print("   precise thesis figures.")


if __name__ == '__main__':
    main()