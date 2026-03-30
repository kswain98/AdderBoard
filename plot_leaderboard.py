#!/usr/bin/env python3
"""
AdderBoard leaderboard visualization.
Single curve with markers colored by category (trained vs hand-coded),
ordered by param count (largest left, smallest right).
Gold stars on the best of each category.
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D

# Hand-coded entries - ordered by params
handcoded = [
    (6, "zcbtrak"),
    (8, "kswain98"),
    (10, "lokimorty"),
    (12, "lokimorty"),
    (20, "yieldthought"),
    (27, "Wonderfall"),
    (28, "jacobli99"),
    (31, "Arch222"),
    (33, "fblissjr"),
    (36, "alexlitz"),
    (50, "lichengliu03"),
    (66, "cosminscn"),
    (87, "bingbangboom-lab"),
    (93, "jacobli99"),
    (111, "corbensorenson"),
    (116, "nino"),
    (121, "Wonderfall"),
    (130, "cosminscn"),
    (130, "Wonderfall"),
    (139, "Wonderfall"),
    (148, "bingbangboom-lab"),
    (177, "xangma"),
    (197, "xangma"),
]

# Trained entries - ordered by params
trained = [
    (36, "tbukic"),
    (39, "lokimorty"),
    (41, "tbukic"),
    (44, "tbukic"),
    (45, "tbukic"),
    (52, "Enara Vijil"),
    (55, "tbukic"),
    (57, "evindor"),
    (58, "tbukic"),
    (62, "tbukic"),
    (67, "evindor"),
    (83, "tbukic"),
    (86, "tbukic"),
    (89, "tbukic"),
    (95, "tbukic"),
    (101, "tbukic"),
    (122, "staghado"),
    (140, "dimopep"),
    (234, "JackCai1206"),
    (262, "lichengliu03"),
    (275, "ryanyord"),
    (305, "h3nock"),
    (311, "rezabyt"),
    (456, "yinglunz"),
    (491, "rezabyt"),
    (512, "yinglunz"),
    (777, "Yeb Havinga"),
    (1644, "anadim"),
    (6080, "anadim"),
]

# --- Color Palette ---
# Deep navy background with warm/cool contrast
BG = '#0f1729'
BG_LIGHT = '#162038'
GRID = '#1e2d4a'
SPINE = '#2a3f6e'
TEXT_PRIMARY = '#e8ecf4'
TEXT_SECONDARY = '#8898bf'
TEXT_MUTED = '#5a6a8f'
LINE_COLOR = '#3a4f7a'

# Teal for hand-coded, coral/orange for trained
HC_COLOR = '#4ecdc4'       # teal
HC_COLOR_LIGHT = '#7eddd6'
TR_COLOR = '#ff6b6b'       # coral
TR_COLOR_LIGHT = '#ff9999'
GOLD = '#ffd700'
GOLD_EDGE = '#c5a000'

# Combine all entries, sort descending (big left, small right)
all_entries = [(p, name, "handcoded") for p, name in handcoded] + \
              [(p, name, "trained") for p, name in trained]
all_entries.sort(key=lambda x: x[0], reverse=True)

# Create figure
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Plot the connecting line
params = [e[0] for e in all_entries]
indices = list(range(1, len(all_entries) + 1))
ax.plot(indices, params, color=LINE_COLOR, linewidth=2, zorder=1, alpha=0.7)

# Plot markers by category
for i, (p, name, cat) in enumerate(all_entries):
    idx = i + 1
    if cat == "handcoded":
        color = HC_COLOR
        marker = 's'
        ms = 14
        edge = HC_COLOR_LIGHT
    else:
        color = TR_COLOR
        marker = 'o'
        ms = 14
        edge = TR_COLOR_LIGHT
    ax.scatter(idx, p, c=color, s=ms**2, marker=marker, zorder=3,
               edgecolors=edge, linewidths=0.8, alpha=0.9)

# Find best (smallest) of each category and their plot positions
best_hc_idx, best_hc_params, best_hc_name = None, None, None
best_tr_idx, best_tr_params, best_tr_name = None, None, None
for i, (p, name, cat) in enumerate(all_entries):
    if cat == "handcoded" and (best_hc_params is None or p < best_hc_params):
        best_hc_idx, best_hc_params, best_hc_name = i + 1, p, name
    if cat == "trained" and (best_tr_params is None or p < best_tr_params):
        best_tr_idx, best_tr_params, best_tr_name = i + 1, p, name

# Stroke effect for labels
stroke = [pe.withStroke(linewidth=4, foreground=BG)]

# --- Both trophies at the same y level (use trained star height) ---
star_y = best_tr_params * 2.5  # common y for both stars
label_y = star_y * 1.6         # "#1 ..." label
name_y = best_tr_params * 1.5  # author name
param_y = best_tr_params * 1.15 # param count

# Best Hand-Coded Trophy
hc_x = best_hc_idx
ax.plot(hc_x, star_y, marker='*', markersize=32, color=GOLD,
        markeredgecolor=GOLD_EDGE, markeredgewidth=1.5, zorder=6)
ax.text(hc_x, label_y, '#1 Hand-coded',
        color=GOLD, fontsize=13, ha='center', va='bottom', fontweight='bold',
        path_effects=stroke, zorder=5)
ax.text(hc_x, name_y, best_hc_name,
        color=HC_COLOR_LIGHT, fontsize=14, ha='center', va='bottom', fontweight='bold',
        path_effects=stroke, zorder=5)
ax.text(hc_x, param_y, f'{best_hc_params}p',
        color=TEXT_SECONDARY, fontsize=16, ha='center', va='bottom', fontweight='bold',
        path_effects=stroke, zorder=5)
ax.plot([hc_x, hc_x], [best_hc_params * 1.05, star_y * 0.85],
        color=GOLD, linewidth=1, alpha=0.4, zorder=4, linestyle='--')

# Best Trained Trophy
tr_x = best_tr_idx
ax.plot(tr_x, star_y, marker='*', markersize=32, color=GOLD,
        markeredgecolor=GOLD_EDGE, markeredgewidth=1.5, zorder=6)
ax.text(tr_x, label_y, '#1 Trained',
        color=GOLD, fontsize=13, ha='center', va='bottom', fontweight='bold',
        path_effects=stroke, zorder=5)
ax.text(tr_x, name_y, best_tr_name,
        color=TR_COLOR_LIGHT, fontsize=14, ha='center', va='bottom', fontweight='bold',
        path_effects=stroke, zorder=5)
ax.text(tr_x, param_y, f'{best_tr_params}p',
        color=TEXT_SECONDARY, fontsize=16, ha='center', va='bottom', fontweight='bold',
        path_effects=stroke, zorder=5)
ax.plot([tr_x, tr_x], [best_tr_params * 1.05, star_y * 0.85],
        color=GOLD, linewidth=1, alpha=0.4, zorder=4, linestyle='--')

# --- Styling ---
ax.set_yscale('log')
ax.set_xlabel('Submission (ordered by parameter count)', fontsize=19,
              color=TEXT_PRIMARY, labelpad=12, fontweight='medium')
ax.set_ylabel('Parameters (log scale)', fontsize=19,
              color=TEXT_PRIMARY, labelpad=12, fontweight='medium')
ax.set_title('AdderBoard: A Benchmark for Minimum-Parameter Addition Transformers',
             fontsize=22, color=TEXT_PRIMARY, fontweight='bold', pad=20)

# Tick styling
ax.tick_params(axis='both', colors='#ffffff', labelsize=16)
ax.spines['bottom'].set_color(SPINE)
ax.spines['left'].set_color(SPINE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.8)
ax.grid(axis='x', color=GRID, linewidth=0.3, alpha=0.3)

# Y-axis ticks
ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
ax.set_yticklabels(['5', '10', '20', '50', '100', '200', '500', '1K', '2K', '5K'])

# X-axis
ax.set_xlim(0, len(all_entries) + 4)

# Add "smaller is better" arrow in bottom-center area (clear space)
ax.annotate('', xy=(0.75, 0.04), xytext=(0.55, 0.04), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color=TEXT_MUTED, lw=2))
ax.text(0.65, 0.07, 'smaller is better', transform=ax.transAxes,
        fontsize=15, color=TEXT_MUTED, ha='center', va='bottom',
        fontstyle='italic')

# Legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=HC_COLOR,
           markersize=13, label='Hand-coded', linestyle='None',
           markeredgecolor=HC_COLOR_LIGHT, markeredgewidth=0.8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TR_COLOR,
           markersize=13, label='Trained', linestyle='None',
           markeredgecolor=TR_COLOR_LIGHT, markeredgewidth=0.8),
]
legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=14,
                   frameon=True, facecolor=BG_LIGHT, edgecolor=SPINE,
                   labelcolor=TEXT_PRIMARY)

# Subtitle
ax.text(0.5, -0.12,
        f'Hand-coded: {len(handcoded)} submissions (best: {handcoded[0][0]}p by {handcoded[0][1]})  |  '
        f'Trained: {len(trained)} submissions (best: {trained[0][0]}p by {trained[0][1]})',
        transform=ax.transAxes, fontsize=15, color=TEXT_SECONDARY,
        ha='center', va='top')

plt.tight_layout()
plt.savefig('/Users/anadim/AdderBoard/leaderboard_race.png', dpi=200,
            bbox_inches='tight', facecolor=BG)
print("Saved to /Users/anadim/AdderBoard/leaderboard_race.png")
plt.close()
