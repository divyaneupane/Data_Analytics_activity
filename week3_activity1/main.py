import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
df = pd.read_csv(r"D:\Master SE\2nd trimister\Data analytics\week3_activity1\age_networth.csv")

# Display first few rows
print("Dataset preview:")
print(df.head())


df.columns = ['Age', 'Net_Worth']

# Calculate correlation
r, p_value = stats.pearsonr(df['Age'], df['Net_Worth'])
print(f"\nPearson correlation coefficient (r) = {r:.4f}")
print(f"p-value = {p_value:.4f}")

# ---------- Visualization ----------
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Net_Worth'], color='steelblue', s=100, edgecolors='black', alpha=0.7)

# Add regression line
m, b = np.polyfit(df['Age'], df['Net_Worth'], 1)
plt.plot(df['Age'], m*df['Age']+b, 'r--', linewidth=2, label=f'Linear fit (r = {r:.2f})')

plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Net Worth ($)', fontsize=12)
plt.title('Correlation between Age and Net Worth', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_age_networth.png', dpi=300)
plt.show()

