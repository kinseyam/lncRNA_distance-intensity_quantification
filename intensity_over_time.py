# Intensity over time

!pip install openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# === CONFIGURATION ===
EXCEL_FILE = 'lncRNA_intensity_timepoints.xlsx'

TIMEPOINTS = {
    'Control': 'Control',
    '12hr': '12hr',
    '18hr': '18hr',
    '24hr': '24hr'
}

# === LOAD DATA FROM ALL TIMEPOINTS ===
print("Loading data from all timepoints...\n")

all_timepoint_data = {}

for sheet_name, display_name in TIMEPOINTS.items():
    try:
        # Read without header to get raw data
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, header=None)

        print(f"Processing {display_name}...")
        print(f"  Sheet dimensions: {df.shape[0]} rows x {df.shape[1]} columns")

        # Each image takes 8 columns: #, Area, Mean, X, Y, empty, empty, empty
        # Mean is at column indices 2, 10, 18, 26, 34... (every 8 columns starting at 2)
        # Row 0 = header row, Row 1 onwards = data

        # Find all Mean column indices (every 8 columns starting at index 2)
        num_cols = df.shape[1]
        mean_col_indices = list(range(2, num_cols, 8))

        print(f"  Found {len(mean_col_indices)} images")

        # Extract all Mean values, skipping header row (row 0) and row 1 (which has column names)
        all_intensities = []

        for col_idx in mean_col_indices:
            if col_idx < num_cols:
                # Skip first 2 rows (header rows), get the Mean column
                col_data = df.iloc[2:, col_idx]  # Skip 2 header rows

                # Convert to numeric, drop NaN and any text
                col_data = pd.to_numeric(col_data, errors='coerce').dropna()

                # Filter out unreasonably small or large values
                col_data = col_data[col_data > 0]

                all_intensities.extend(col_data.values)

        if len(all_intensities) == 0:
            print(f"  Warning: No data found for {display_name}")
            continue

        intensities = np.array(all_intensities)
        all_timepoint_data[display_name] = intensities

        print(f"  Total measurements: {len(intensities)}")
        print(f"  Range: {intensities.min():.1f} - {intensities.max():.1f}")
        print(f"  Mean: {intensities.mean():.1f} ± {intensities.std():.1f}\n")

    except Exception as e:
        print(f"Error loading {sheet_name}: {e}")
        import traceback
        traceback.print_exc()

if len(all_timepoint_data) == 0:
    print("\nNo data loaded! Check sheet names:")
    excel_file = pd.ExcelFile(EXCEL_FILE)
    print(f"Available sheets: {excel_file.sheet_names}")
else:
    print(f"✓ Successfully loaded {len(all_timepoint_data)} timepoints")

print("\n" + "="*80)

# === COMPUTE STATISTICS ===
for timepoint, intensities in all_timepoint_data.items():
    mu = intensities.mean()
    sigma = intensities.std()
    sem = sigma / np.sqrt(len(intensities))
    print(f"{timepoint}: μ={mu:.1f} ± {sem:.1f} AU (SEM), n={len(intensities)}")

print("\n" + "="*80)

# === CREATE OVERLAY PLOT ===
fig, ax = plt.subplots(figsize=(12, 8))

timepoint_colors = {
    'Control': 'black',
    '12hr': 'blue',
    '18hr': 'green',
    '24hr': 'red'
}

# Find global x-axis range
all_vals = np.concatenate([i for i in all_timepoint_data.values()])
x_min = all_vals.min()
x_max = all_vals.max()
x = np.linspace(x_min, x_max, 500)

legend_handles = []
legend_labels = []

# Plot in order
plot_order = ['Control', '12hr', '18hr', '24hr']

for timepoint in plot_order:
    if timepoint not in all_timepoint_data:
        continue

    intensities = all_timepoint_data[timepoint]
    mu = intensities.mean()
    sigma = intensities.std()
    sem = sigma / np.sqrt(len(intensities))

    # Generate normal distribution curve
    y = stats.norm.pdf(x, mu, sigma)

    color = timepoint_colors.get(timepoint, 'gray')

    line, = ax.plot(x, y, linewidth=3, color=color)
    ax.axvline(mu, color=color, linestyle='--', alpha=0.3, linewidth=1.5)
    ax.axvspan(mu - sem, mu + sem, alpha=0.12, color=color)

    legend_handles.append(line)
    legend_labels.append(timepoint)

# Formatting
ax.set_xlabel('Intensity (AU)', fontsize=16, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=16, fontweight='bold')

ax.grid(True, alpha=0.3)
ax.legend(legend_handles, legend_labels,
          loc='upper right',
          fontsize=14,
          framealpha=0.95,
          title='Timepoint',
          title_fontsize=15)

plt.tight_layout()
plt.savefig('lncRNA_intensity_timepoints.pdf', bbox_inches='tight', format='pdf')
plt.show()

print("\n✓ Saved as 'lncRNA_intensity_timepoints.pdf'")

# === SUMMARY TABLE ===
summary_data = []
for timepoint in plot_order:
    if timepoint not in all_timepoint_data:
        continue
    intensities = all_timepoint_data[timepoint]
    mu = intensities.mean()
    sigma = intensities.std()
    sem = sigma / np.sqrt(len(intensities))

    summary_data.append({
        'Timepoint': timepoint,
        'Mean (AU)': f"{mu:.2f}",
        'SEM (AU)': f"{sem:.2f}",
        'Std Dev (AU)': f"{sigma:.2f}",
        'N (total puncta)': len(intensities)
    })

summary_df = pd.DataFrame(summary_data)
print("\n=== SUMMARY TABLE ===")
print(summary_df.to_string(index=False))
summary_df.to_csv('lncRNA_intensity_timepoints_summary.csv', index=False)
print("\n✓ Saved to 'lncRNA_intensity_timepoints_summary.csv'")

# === TEMPORAL CHANGES ===
print("\n=== TEMPORAL CHANGES vs CONTROL ===")
if 'Control' in all_timepoint_data:
    control_mean = all_timepoint_data['Control'].mean()
    for timepoint in ['12hr', '18hr', '24hr']:
        if timepoint in all_timepoint_data:
            tp_mean = all_timepoint_data[timepoint].mean()
            change = ((tp_mean - control_mean) / control_mean) * 100
            print(f"{timepoint} vs Control: {change:+.1f}% change in mean intensity")