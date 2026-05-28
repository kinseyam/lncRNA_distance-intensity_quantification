# Distance by patient

# Install required library for Excel
!pip install openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import stats
from itertools import combinations

# === CONFIGURATION ===
EXCEL_FILE = 'lncRNA_distances_patients.xlsx'
PIXEL_SIZE_UM = 0.102  # micrometers per pixel
MAX_DISTANCE_UM = 5  # Adjust if needed
MAX_DISTANCE_PIXELS = MAX_DISTANCE_UM / PIXEL_SIZE_UM

# === GET ALL SHEET NAMES ===
excel_file = pd.ExcelFile(EXCEL_FILE)
sheet_names = excel_file.sheet_names
print(f"Found {len(sheet_names)} sheets: {sheet_names}\n")

# === PROCESS EACH SHEET - COMBINE ALL COLOR PAIRS ===
sheet_stats = {}

for sheet_idx, sheet_name in enumerate(sheet_names):
    print(f"Processing {sheet_name}...")
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)

    # Extract coordinates for all colors
    colors_dict = {
        'green': df[['X', 'Y']].dropna().values,
        'red': df[['X.1', 'Y.1']].dropna().values,
        'yellow': df[['X.2', 'Y.2']].dropna().values,
        'blue': df[['X.3', 'Y.3']].dropna().values,
        'black': df[['X.4', 'Y.4']].dropna().values
    }

    # Collect all distances from all color pairs for this sheet
    all_distances_this_sheet = []

    # Calculate distances for all color combinations
    color_pairs = list(combinations(colors_dict.keys(), 2))

    for color1, color2 in color_pairs:
        coords1 = colors_dict[color1]
        coords2 = colors_dict[color2]

        if len(coords1) == 0 or len(coords2) == 0:
            continue

        # Calculate pairwise distances
        pairwise_distances = cdist(coords1, coords2)
        min_distances = pairwise_distances.min(axis=1)
        valid_distances_pixels = min_distances[min_distances < MAX_DISTANCE_PIXELS]
        valid_distances_um = valid_distances_pixels * PIXEL_SIZE_UM

        # Add to this sheet's combined distances
        all_distances_this_sheet.extend(valid_distances_um)

    if len(all_distances_this_sheet) == 0:
        print(f"  Skipping {sheet_name}: no valid distances")
        continue

    # Convert to numpy array
    all_distances_this_sheet = np.array(all_distances_this_sheet)

    # Calculate statistics for this sheet (all color pairs combined)
    mu = all_distances_this_sheet.mean()
    sigma = all_distances_this_sheet.std()
    sem = sigma / np.sqrt(len(all_distances_this_sheet))

    # Store with "Patient" label
    patient_name = f"Patient {sheet_idx + 1}"
    sheet_stats[patient_name] = {
        'mean': mu,
        'std': sigma,
        'sem': sem,
        'n': len(all_distances_this_sheet),
        'distances': all_distances_this_sheet
    }

    print(f"  {patient_name}: μ={mu:.3f}±{sem:.3f} μm (n={len(all_distances_this_sheet)} pairs)")

# === CREATE OVERLAY PLOT WITH EXTERNAL LEGEND ===
fig, ax = plt.subplots(figsize=(14, 8))

# Color scheme for different patients
colors = plt.cm.tab10(np.linspace(0, 1, len(sheet_stats)))

# Generate x-axis range
all_means = [s['mean'] for s in sheet_stats.values()]
all_stds = [s['std'] for s in sheet_stats.values()]
x_min = min([m - 3*s for m, s in zip(all_means, all_stds)])
x_max = max([m + 3*s for m, s in zip(all_means, all_stds)])
x = np.linspace(max(0, x_min), x_max, 500)

# Plot normal distribution curve for each patient
legend_handles = []
legend_labels = []

for idx, (patient_name, stats_dict) in enumerate(sheet_stats.items()):
    mu = stats_dict['mean']
    sigma = stats_dict['std']
    sem = stats_dict['sem']

    # Generate normal distribution curve
    y = stats.norm.pdf(x, mu, sigma)

    # Plot the curve
    line, = ax.plot(x, y, linewidth=2.5, color=colors[idx])

    # Add vertical line at mean
    ax.axvline(mu, color=colors[idx], linestyle='--', alpha=0.3, linewidth=1.5)

    # Add shaded region for ±SEM
    ax.axvspan(mu - sem, mu + sem, alpha=0.15, color=colors[idx])

    # Store for legend
    legend_handles.append(line)
    legend_labels.append(f'{patient_name}')

# Calculate overall mean and SEM across all patients
overall_mean = np.mean(all_means)
overall_sem = np.std(all_means) / np.sqrt(len(all_means))

# Add overall mean line
overall_line = ax.axvline(overall_mean, color='black', linestyle='-', linewidth=3, alpha=0.7)
ax.axvspan(overall_mean - overall_sem, overall_mean + overall_sem,
           alpha=0.2, color='black')

legend_handles.append(overall_line)
legend_labels.append(f'Overall Mean')

# Formatting
ax.set_xlabel('Distance (μm)', fontsize=16, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Place legend outside the plot on the right
ax.legend(legend_handles, legend_labels,
          loc='center left',
          bbox_to_anchor=(1.02, 0.5),
          fontsize=12,
          framealpha=0.9,
          title='Patient Statistics',
          title_fontsize=13)

plt.tight_layout()
plt.savefig('lncRNA_patient_dist_overlay.pdf', bbox_inches='tight', format='pdf')
plt.show()

print(f"\n✓ Overlay plot saved as 'all_colors_overlay_per_patient.png'")

# === CREATE SUMMARY TABLE ===
summary_data = []
for patient_name, stats_dict in sheet_stats.items():
    summary_data.append({
        'Patient': patient_name,
        'Mean (μm)': f"{stats_dict['mean']:.3f}",
        'SEM (μm)': f"{stats_dict['sem']:.3f}",
        'Std Dev (μm)': f"{stats_dict['std']:.3f}",
        'N pairs': stats_dict['n']
    })

# Add overall statistics
summary_data.append({
    'Patient': 'OVERALL',
    'Mean (μm)': f"{overall_mean:.3f}",
    'SEM (μm)': f"{overall_sem:.3f}",
    'Std Dev (μm)': f"{np.std(all_means):.3f}",
    'N patients': len(sheet_stats)
})

summary_df = pd.DataFrame(summary_data)
print("\n=== SUMMARY TABLE ===")
print(summary_df.to_string(index=False))

summary_df.to_csv('all_colors_per_patient_summary.csv', index=False)
print(f"\n✓ Summary saved to 'all_colors_per_patient_summary.csv'")

# === VARIABILITY ANALYSIS ===
cv = (np.std(all_means) / overall_mean) * 100
print("\n=== VARIABILITY ANALYSIS ===")
print(f"Coefficient of Variation: {cv:.1f}%")
print(f"Mean distance across all patients: {overall_mean:.3f} ± {overall_sem:.3f} μm")
if cv < 30:
    print("✓ Low variability across patients - distances are consistent!")
else:
    print("⚠ High variability across patients - consider more measurements")