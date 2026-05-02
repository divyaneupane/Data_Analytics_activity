import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"D:\Master SE\2nd trimister\Data analytics\week3_activity2\messy_dataset_Mukesh.csv")
print("Original data:")
print(df)
print("\nMissing values:\n", df.isnull().sum())


# Duplicate IDs (excluding NaN)
dups = df[df['ID'].notna() & df.duplicated('ID', keep=False)]
print(f"\nDuplicate ID rows: {len(dups)}")
print(dups[['ID', 'Name']])

# Text in numeric columns
print("\nNon-numeric Age values:")
print(df[~df['Age'].astype(str).str.isdigit() & df['Age'].notna()]['Age'].unique())

print("\nNon-numeric Salary values:")
print(df[~df['Salary'].astype(str).str.replace(',','').str.isdigit() & df['Salary'].notna()]['Salary'].unique())

# Invalid dates (after attempt to parse)
df['Join Date_temp'] = pd.to_datetime(df['Join Date'], errors='coerce', dayfirst=True)
print(f"\nInvalid dates: {df['Join Date_temp'].isna().sum()} rows")
print(df[df['Join Date_temp'].isna()][['ID', 'Name', 'Join Date']])


# 2. CLEANING – fix Age
age_list = []
for a in df['Age']:
    if pd.isna(a):
        age_list.append(np.nan)
    else:
        a_str = str(a).lower()
        if "thirty" in a_str and "eight" in a_str:
            age_list.append(38)
        else:
            try:
                age_list.append(int(a_str))
            except:
                age_list.append(np.nan)
df['Age'] = age_list


# 3. CLEANING – fix Salary

salary_list = []
for s in df['Salary']:
    if pd.isna(s):
        salary_list.append(np.nan)
    else:
        s_str = str(s).lower()
        if "sixty five thousand" in s_str:
            salary_list.append(65000)
        else:
            digits = [int(ch) for ch in s_str.split() if ch.isdigit()]
            if digits:
                salary_list.append(digits[0])
            else:
                salary_list.append(np.nan)
df['Salary'] = salary_list


# 4. Fix ID

df['ID'] = pd.to_numeric(df['ID'], errors='coerce')


# 5. Merge duplicate ID=2 (keep first non-null)

df = df.groupby("ID").first().reset_index(drop=True)  # 


# 6. Fix dates (column name is 'Join Date')

df['Join Date'] = pd.to_datetime(df['Join Date'], errors='coerce', dayfirst=True)


# 7. Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())          # no inplace=True
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
df['Country'] = df['Country'].fillna("NZ")                
df['Name'] = df['Name'].fillna("Unknown")
df['Join Date'] = df['Join Date'].fillna(df['Join Date'].median())

# 8. Convert AUD to NZD (1.08 rate)

df["Salary_NZD"] = df["Salary"]
for idx in df.index:
    if df.loc[idx, "Country"] == "AU":
        df.loc[idx, "Salary_NZD"] = df.loc[idx, "Salary"] * 1.08

# 9. Show cleaned data
print("\n=== CLEANED DATA ===")
print(df)
print("\nMissing after cleaning:\n", df.isnull().sum())

# 10. Pearson correlation heatmap

numeric_cols = ["Age", "Salary_NZD"]
corr_matrix = df[numeric_cols].corr(method="pearson")
print("\n=== PEARSON CORRELATION ===")
print(corr_matrix)

plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Pearson Correlation Heatmap\n(Age vs Salary_NZD)")
plt.savefig("heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Heatmap saved as heatmap.png")


# 11. Outlier detection (IQR)

print("\n=== OUTLIER REPORT (IQR) ===")
for col in ["Age", "Salary_NZD"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"\n{col}:")
    print(f"  Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}")
    print(f"  Lower={lower:.1f}, Upper={upper:.1f}")
    if outliers.empty:
        print("  → No outliers detected.")
    else:
        print("  → Outliers found:")
        print(outliers[["ID", "Name", col]])


# 12. Save cleaned CSV

df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned data saved to cleaned_dataset.csv")


# 13. Bar graph: Mean vs Median for Age and Salary_NZD
print("\n=== GENERATING BAR GRAPH (MEAN vs MEDIAN) ===")

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Data for Age
age_mean = df['Age'].mean()
age_median = df['Age'].median()
age_labels = ['Mean', 'Median']
age_values = [age_mean, age_median]
age_colors = ['#1f77b4', '#ff7f0e']

# Data for Salary_NZD
salary_mean = df['Salary_NZD'].mean()
salary_median = df['Salary_NZD'].median()
salary_labels = ['Mean', 'Median']
salary_values = [salary_mean, salary_median]
salary_colors = ['#2ca02c', '#d62728']

# Plot Age bar chart
ax1.bar(age_labels, age_values, color=age_colors, edgecolor='black')
ax1.set_ylabel('Years')
ax1.set_title('Age: Mean vs Median')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
# Add value labels on bars
for i, v in enumerate(age_values):
    ax1.text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

# Plot Salary_NZD bar chart
ax2.bar(salary_labels, salary_values, color=salary_colors, edgecolor='black')
ax2.set_ylabel('NZD')
ax2.set_title('Salary_NZD: Mean vs Median')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
# Add value labels on bars
for i, v in enumerate(salary_values):
    ax2.text(i, v + 1000, f'${v:,.0f}', ha='center', fontweight='bold')

# Overall title
fig.suptitle('Comparison of Mean and Median Values (After Cleaning)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('mean_median_bars.png', dpi=150, bbox_inches='tight')
plt.show()
print("Bar graph saved as mean_median_bars.png")


# 14. Boxplot to visualize outliers (or absence of outliers)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Boxplot for Age
ax1.boxplot(df['Age'], vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
ax1.set_title('Age Distribution (No Outliers)')
ax1.set_ylabel('Years')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Boxplot for Salary_NZD
ax2.boxplot(df['Salary_NZD'], vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
ax2.set_title('Salary_NZD Distribution (No Outliers)')
ax2.set_ylabel('NZD')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Boxplot with IQR Fences – No Outliers Detected', fontsize=14)
plt.tight_layout()
plt.savefig('boxplot_no_outliers.png', dpi=150)
plt.show()
print("Boxplot saved as boxplot_no_outliers.png")