# Distance over time

# Install required library for Excel
!pip install openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import stats
from itertools import combinations

# === CONFIGURATION ===
EXCEL_FILE = 'distance_quantification_timepoints.xlsx'
MAX_DISTANCE_UM = 5  # Maximum distance in µm
PIXEL_SIZE_UM = 0.102  # Pixel to µm conversion for treatment sheets

# Define which sheets are already in µm vs pixels
SHEET_CONFIG = {
    'Control': {'display': 'Control', 'already_um': False},
    '12 hr':   {'display': '12hr',    'already_um': False},
    '18 hr':   {'display': '18hr',    'already_um': False},
    '24 hr':   {'display': '24hr',    'already_um': False},
}

# === GET ALL SHEET NAMES ===
excel_file = pd.ExcelFile(EXCEL_FILE)
sheet_names = excel_file.sheet_names
print(f"Found {len(sheet_names)} sheets: {sheet_names}\n")

def extract_coords(df, x_col, y_col, pixel_size):
    """Extract numeric X,Y coordinates and convert to µm"""
    data = df[[x_col, y_col]].copy()
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    coords = data.values * pixel_size  # Convert to µm (or keep as µm if already converted)
    return coords

# === PROCESS EACH SHEET ===
timepoint_stats = {}

for sheet_name, config in SHEET_CONFIG.items():
    display_name = config['display']
    already_um = config['already_um']

    # Set pixel size: 1.0 if already in µm, otherwise convert
    pixel_size = 1.0 if already_um else PIXEL_SIZE_UM

    print(f"Processing {sheet_name} (pixel size: {pixel_size})...")
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, header=1)

    # Find all X and Y column pairs
    all_cols = df.columns.tolist()
    x_cols = [col for col in all_cols if str(col) == 'X' or str(col).startswith('X.')]
    y_cols = [col for col in all_cols if str(col) == 'Y' or str(col).startswith('Y.')]

    print(f"  Found {len(x_cols)} X columns and {len(y_cols)} Y columns")

    # Extract coordinates for all 5 colors
    color_names = ['green', 'red', 'yellow', 'blue', 'gray']
    colors_dict = {}

    for i, color in enumerate(color_names):
        if i < len(x_cols) and i < len(y_cols):
            coords = extract_coords(df, x_cols[i], y_cols[i], pixel_size)
            colors_dict[color] = coords
            if len(coords) > 0:
                print(f"    {color}: {len(coords)} spots, "
                      f"X range: {coords[:,0].min():.2f}-{coords[:,0].max():.2f} µm")

    # Collect all distances from all color pairs
    all_distances = []
    color_pairs = list(combinations(color_names, 2))

    for color1, color2 in color_pairs:
        coords1 = colors_dict.get(color1, np.array([]))
        coords2 = colors_dict.get(color2, np.array([]))

        if len(coords1) == 0 or len(coords2) == 0:
            continue

        # Calculate pairwise Euclidean distances (in µm)
        pairwise_distances = cdist(coords1, coords2)
        min_distances = pairwise_distances.min(axis=1)

        # Filter to max distance threshold
        valid_distances = min_distances[min_distances < MAX_DISTANCE_UM]
        all_distances.extend(valid_distances)

    if len(all_distances) == 0:
        print(f"  Skipping {display_name}: no valid distances found")
        continue

    all_distances = np.array(all_distances)
    mu = all_distances.mean()
    sigma = all_distances.std()
    sem = sigma / np.sqrt(len(all_distances))

    timepoint_stats[display_name] = {
        'mean': mu,
        'std': sigma,
        'sem': sem,
        'n': len(all_distances),
        'distances': all_distances
    }

    print(f"  → {display_name}: μ={mu:.3f}±{sem:.3f} µm (n={len(all_distances)} pairs)\n")

print("=" * 80)

# === CREATE OVERLAY PLOT ===
fig, ax = plt.subplots(figsize=(12, 8))

timepoint_colors = {
    'Control': 'black',
    '12hr': 'blue',
    '18hr': 'green',
    '24hr': 'red'
}

# Generate x-axis range
all_means = [s['mean'] for s in timepoint_stats.values()]
all_stds = [s['std'] for s in timepoint_stats.values()]
x_min = min([m - 3*s for m, s in zip(all_means, all_stds)])
x_max = max([m + 3*s for m, s in zip(all_means, all_stds)])
x = np.linspace(max(0, x_min), x_max, 500)

legend_handles = []
legend_labels = []

# Plot in specific order: Control first, then timepoints
plot_order = ['Control', '12hr', '18hr', '24hr']

for timepoint_name in plot_order:
    if timepoint_name not in timepoint_stats:
        continue

    stats_dict = timepoint_stats[timepoint_name]
    mu = stats_dict['mean']
    sigma = stats_dict['std']
    sem = stats_dict['sem']

    y = stats.norm.pdf(x, mu, sigma)
    color = timepoint_colors.get(timepoint_name, 'gray')

    line, = ax.plot(x, y, linewidth=3, color=color)
    ax.axvline(mu, color=color, linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvspan(mu - sem, mu + sem, alpha=0.12, color=color)

    legend_handles.append(line)
    legend_labels.append(timepoint_name)

    print(f"{timepoint_name}: μ={mu:.3f} ± {sem:.3f} µm (n={stats_dict['n']})")

# Formatting
ax.set_xlabel('Distance (μm)', fontsize=16, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=16, fontweight='bold')
ax.set_title('Inter-lncRNA Distance Distributions Over Time',
             fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(legend_handles, legend_labels,
          loc='upper right',
          fontsize=14,
          framealpha=0.95,
          title='Timepoint',
          title_fontsize=15)

plt.tight_layout()
plt.savefig('lncRNA_distance_timepoints.pdf', bbox_inches='tight', format='pdf')
plt.show()

print(f"\n✓ Saved as 'lncRNA_distance_timepoints.pdf'")

# === SUMMARY TABLE ===
summary_data = []
for timepoint_name in plot_order:
    if timepoint_name not in timepoint_stats:
        continue
    stats_dict = timepoint_stats[timepoint_name]
    summary_data.append({
        'Timepoint': timepoint_name,
        'Mean (μm)': f"{stats_dict['mean']:.3f}",
        'SEM (μm)': f"{stats_dict['sem']:.3f}",
        'Std Dev (μm)': f"{stats_dict['std']:.3f}",
        'N pairs': stats_dict['n']
    })

summary_df = pd.DataFrame(summary_data)
print("\n=== SUMMARY TABLE ===")
print(summary_df.to_string(index=False))
summary_df.to_csv('distance_timepoints_summary.csv', index=False)
print("\n✓ Saved to 'distance_timepoints_summary.csv'")

# === TEMPORAL CHANGES ===
print("\n=== TEMPORAL CHANGES ===")
if 'Control' in timepoint_stats:
    control_mean = timepoint_stats['Control']['mean']
    for timepoint_name in ['12hr', '18hr', '24hr']:
        if timepoint_name in timepoint_stats:
            change = ((timepoint_stats[timepoint_name]['mean'] - control_mean) / control_mean) * 100
            print(f"{timepoint_name} vs Control: {change:+.1f}% change in mean distance")