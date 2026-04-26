import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

# ============================================
# 1. LOAD AND COMBINE ALL CSV FILES
# ============================================

path = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(path, "*.csv"))
print(f"Found {len(files)} CSV files in {path}")

if not files:
    raise FileNotFoundError(f"No CSV files found in {path}")

df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

# Clean station column immediately
if 'station' in df.columns:
    df['station'] = df['station'].astype(str)
    df = df[df['station'] != 'nan']
else:
    print("ERROR: 'station' column not found.")
    exit()

# Task 1
print("\nFIRST 5 ROWS:")
print(df.head())

print("\nCOLUMN NAMES & DATA TYPES:")
print(df.dtypes)

print(f"\nTOTAL ROWS: {df.shape[0]:,}")
print(f"TOTAL COLUMNS: {df.shape[1]}")

# 
# 2. HANDLE MISSING VALUES


print("TASK 2: MISSING VALUES")


missing = df.isnull().sum()
print(missing[missing > 0])

numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

essential_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
existing_essential = [c for c in essential_pollutants if c in df.columns]

if existing_essential:
    df.dropna(subset=existing_essential, how='all', inplace=True)

print(f"\nAfter cleaning: {len(df):,} rows remaining")

# 3. OVERALL STATISTICS
# 

print("TASK 3: OVERALL PM2.5 STATISTICS")

if 'PM2.5' in df.columns:
    pm25 = df['PM2.5']
    print(f"Mean:     {pm25.mean():.2f} ug/m³")
    print(f"Median:   {pm25.median():.2f} ug/m³")
    print(f"Min:      {pm25.min():.2f} ug/m³")
    print(f"Max:      {pm25.max():.2f} ug/m³")
    print(f"Std Dev:  {pm25.std():.2f} ug/m³")
else:
    print("Column 'PM2.5' not found.")
    exit()

# 4. AVERAGE POLLUTION PER STATION

print("TASK 4: AVERAGE POLLUTION PER STATION")

all_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
existing_pollutants = [p for p in all_pollutants if p in df.columns]

if not existing_pollutants:
    print("ERROR: No pollutant columns found.")
    exit()

station_avg = df.groupby('station')[existing_pollutants].mean().round(2)
print(station_avg)

if not station_avg.empty and 'PM2.5' in station_avg.columns:
    worst_station = station_avg['PM2.5'].idxmax()
    worst_value = station_avg['PM2.5'].max()
    print(f"\nWorst station (highest PM2.5): {worst_station} ({worst_value:.2f} ug/m³)")

# 5. USER SELECTION (FIXED)

stations = sorted(df['station'].unique())  # Now safe because we cleaned earlier

print("\nSelect station for detailed analysis:")
print("0. All stations (combined)")
for i, name in enumerate(stations, 1):
    print(f"{i}. {name}")

choice = input("\nEnter number: ").strip()

if choice == '0':
    selected = df
    label = "All Stations"
elif choice.isdigit() and 1 <= int(choice) <= len(stations):
    station_name = stations[int(choice)-1]
    selected = df[df['station'] == station_name]
    label = station_name
else:
    print("Invalid choice. Using all stations.")
    selected = df
    label = "All Stations"

print(f"\n--- Selected: {label} ---")
print(f"Records: {len(selected):,}")

# 6. STATISTICS FOR SELECTED STATION

if 'PM2.5' in selected.columns:
    print(f"\nPM2.5 Statistics for {label}:")
    print(f"Mean:   {selected['PM2.5'].mean():.2f} ug/m³")
    print(f"Median: {selected['PM2.5'].median():.2f} ug/m³")
    print(f"Min:    {selected['PM2.5'].min():.2f} ug/m³")
    print(f"Max:    {selected['PM2.5'].max():.2f} ug/m³")
    print(f"Std:    {selected['PM2.5'].std():.2f} ug/m³")

overall_mean = df['PM2.5'].mean()
diff = selected['PM2.5'].mean() - overall_mean
print(f"\nOverall mean PM2.5: {overall_mean:.2f} ug/m³")
if diff > 0:
    print(f"{label} is {diff:.2f} ug/m³ ABOVE the overall average.")
elif diff < 0:
    print(f"{label} is {abs(diff):.2f} ug/m³ BELOW the overall average.")
else:
    print(f"{label} equals the overall average.")


# 7. VISUALIZATIONS

print("TASK 5: GENERATING CHARTS")

if 'PM2.5' in selected.columns:
    plt.figure(figsize=(10,5))
    selected['PM2.5'].hist(bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(selected['PM2.5'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean ({selected["PM2.5"].mean():.1f})')
    plt.axvline(selected['PM2.5'].median(), color='green', linestyle='--', linewidth=2, label=f'Median ({selected["PM2.5"].median():.1f})')
    plt.title(f'PM2.5 Distribution - {label}')
    plt.xlabel('PM2.5 (ug/m³)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('histogram_pm25.png')
    plt.show()

if 'PM2.5' in selected.columns:
    plt.figure(figsize=(12,4))
    selected['PM2.5'].iloc[:1000].plot(color='tomato', linewidth=0.5)
    plt.title(f'PM2.5 Trend (First 1000 Readings) - {label}')
    plt.xlabel('Sample Index')
    plt.ylabel('PM2.5 (ug/m³)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lineplot_pm25.png')
    plt.show()

if len(existing_pollutants) >= 1:
    plt.figure(figsize=(10,6))
    selected[existing_pollutants].plot(kind='box')
    plt.title(f'Pollutant Distribution - {label}')
    plt.ylabel('Concentration (ug/m³)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('boxplot_pollutants.png')
    plt.show()


# 8. CORRELATION ANALYSIS

print("TASK 6: CORRELATION ANALYSIS")

corr_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
existing_corr = [c for c in corr_cols if c in selected.columns]

if len(existing_corr) >= 2:
    corr_matrix = selected[existing_corr].corr()
    if 'PM2.5' in corr_matrix.columns:
        print("\nCorrelation of PM2.5 with other variables:")
        corr_pm25 = corr_matrix['PM2.5'].sort_values(ascending=False)
        print(corr_pm25.round(3))
        if len(corr_pm25) > 1:
            most_correlated = corr_pm25.index[1]
            print(f"\nMost correlated with PM2.5: {most_correlated} (r={corr_pm25.iloc[1]:.3f})")

    try:
        import seaborn as sns
        plt.figure(figsize=(10,8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
        plt.title(f'Correlation Heatmap - {label}')
        plt.tight_layout()
        plt.savefig('heatmap.png')
        plt.show()
    except ImportError:
        print("\nSeaborn not installed. Skipping heatmap.")
        print("Install with: pip install seaborn")
else:
    print("Not enough numeric columns for correlation analysis.")

print("ANALYSIS COMPLETE")





































