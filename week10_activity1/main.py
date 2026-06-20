import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_csv(r"D:\Master SE\2nd trimister\Data analytics\week10_activity1\salary-dataset.csv")
# Display first few rows
print("Dataset preview:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for negative values in numeric columns
print("\nNegative values in numeric columns:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        print(f"{col}: {neg_count} negative values")
    else:
        print(f"{col}: No negative values")

# Check for duplicates in YearsExperience
print("\nChecking for duplicates in YearsExperience...")
duplicate_count = df['YearsExperience'].duplicated().sum()
print(f"Duplicate values in YearsExperience: {duplicate_count}")

if duplicate_count > 0:
    # Store original for reference
    df_original = df.copy()
    
    # Take MEAN salary for same years
    df = df.groupby('YearsExperience', as_index=False).agg({
        'Salary': 'mean'
    })
    
    print(f"   Original rows: {len(df_original)}")
    print(f"   New rows: {len(df)}")
    print(f"   Rows removed: {len(df_original) - len(df)}")

# Sort by experience
df = df.sort_values('YearsExperience').reset_index(drop=True)

print(f"\nShape after cleaning: {df.shape}")
print("\nFirst 10 rows after cleaning:")
print(df.head(10))

print(f"\nSummary statistics:")
print(df.describe())

# LINEAR REGRESSION


print("LINEAR REGRESSION MODEL")


X = df[['YearsExperience']].values
y = df[['Salary']].values

# Split data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae_val = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Get coefficients
intercept = float(model.intercept_[0] if isinstance(model.intercept_, np.ndarray) else model.intercept_)
coef = float(model.coef_[0][0] if isinstance(model.coef_, np.ndarray) and len(model.coef_.shape) > 1 else model.coef_[0])

print(f"\nModel Equation: Salary = {intercept:.2f} + {coef:.2f} * YearsExperience")
print(f"Model Intercept: {intercept:.2f}")
print(f"Model Coefficient (Slope): {coef:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"Mean Absolute Error (MAE): ${mae_val:,.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# POLYNOMIAL REGRESSION (degree=2)

print("POLYNOMIAL REGRESSION MODEL (degree=2)")


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split polynomial features into 70% training and 30% testing
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.3, random_state=42)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_poly)
y_pred_poly = model_poly.predict(X_test_poly)

# Calculate polynomial metrics
mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
r2_poly = r2_score(y_test_poly, y_pred_poly)
mae_poly = mean_absolute_error(y_test_poly, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)

# Get polynomial coefficients
poly_intercept = float(model_poly.intercept_[0] if isinstance(model_poly.intercept_, np.ndarray) else model_poly.intercept_)
poly_coefs = model_poly.coef_[0] if isinstance(model_poly.coef_, np.ndarray) and len(model_poly.coef_.shape) > 1 else model_poly.coef_

print(f"\nModel Equation: Salary = {poly_intercept:.2f} + {poly_coefs[1]:.2f}*X + {poly_coefs[2]:.2f}*X^2")
print(f"Model Intercept: {poly_intercept:.2f}")
print(f"Model Coefficients: {poly_coefs}")
print(f"Mean Squared Error (MSE): {mse_poly:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse_poly:,.2f}")
print(f"Mean Absolute Error (MAE): ${mae_poly:,.2f}")
print(f"R-squared (R2) Score: {r2_poly:.4f} ({r2_poly*100:.2f}%)")

# PREDICT SALARY FOR 14, 14.5, AND 15 YEARS

print("SALARY PREDICTIONS FOR 14, 14.5, AND 15 YEARS OF EXPERIENCE")

# Create array with experience values to predict
experience_to_predict = np.array([[14], [14.5], [15]])

# Linear Regression Predictions
linear_predictions = model.predict(experience_to_predict)

# Polynomial Regression Predictions (degree=2)
poly_predictions = model_poly.predict(poly.transform(experience_to_predict))

# Display results in a table
print("\n------------------------------------------------------------")
print("Years Experience   Linear Model      Polynomial Model")
print("------------------------------------------------------------")

for i, exp in enumerate([14, 14.5, 15]):
    linear_val = linear_predictions[i][0] if isinstance(linear_predictions[i], np.ndarray) else linear_predictions[i]
    poly_val = poly_predictions[i][0] if isinstance(poly_predictions[i], np.ndarray) else poly_predictions[i]
    print(f"{exp:18.1f}  ${linear_val:14,.2f}     ${poly_val:18,.2f}")



# EXISTING DATA NEAR PREDICTION RANGE

print("EXISTING DATA NEAR 13+ YEARS EXPERIENCE (for reference):")


nearby_exp = df[df['YearsExperience'] >= 12]
if not nearby_exp.empty:
    print(nearby_exp.to_string(index=False))
else:
    print("No data available for 12+ years")

max_exp = df['YearsExperience'].max()
print(f"\nMaximum experience in dataset: {max_exp} years")
print("NOTE: Predictions for 14-15 years are EXTRAPOLATIONS beyond the training data range!")


# VISUALIZATION
# Visualization
X_range = np.linspace(X.min(), X.max() + 2, 200).reshape(-1, 1)
y_range_linear = model.predict(X_range)
y_range_poly = model_poly.predict(poly.transform(X_range))

plt.figure(figsize=(12, 7))

plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Data', s=80)
plt.plot(X_range, y_range_linear, color='green', linewidth=2, label='Linear Regression')
plt.plot(X_range, y_range_poly, color='red', linewidth=2, label='Polynomial (degree=2)')

plt.scatter(experience_to_predict, linear_predictions, 
           color='green', s=200, marker='s', 
           edgecolors='darkgreen', linewidth=2,
           label='Linear Predictions')

plt.scatter(experience_to_predict, poly_predictions, 
           color='red', s=200, marker='^', 
           edgecolors='darkred', linewidth=2,
           label='Polynomial Predictions')

for i, exp in enumerate([14, 14.5, 15]):
    linear_val = linear_predictions[i][0] if isinstance(linear_predictions[i], np.ndarray) else linear_predictions[i]
    poly_val = poly_predictions[i][0] if isinstance(poly_predictions[i], np.ndarray) else poly_predictions[i]
    
    plt.annotate(f'${linear_val:,.0f}', 
                xy=(exp, linear_val),
                xytext=(exp+0.3, linear_val+5000),
                fontsize=9, color='darkgreen', weight='bold')
    
    plt.annotate(f'${poly_val:,.0f}', 
                xy=(exp, poly_val),
                xytext=(exp+0.3, poly_val-8000),
                fontsize=9, color='darkred', weight='bold')

plt.axvline(x=df['YearsExperience'].max(), color='gray', linestyle='--', alpha=0.5, 
            label=f'Max Data: {df["YearsExperience"].max()} yrs')

plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)
plt.title('Salary Prediction: Linear vs Polynomial Regression', fontsize=14, weight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 18, 2))
plt.tight_layout()
plt.show()

# Model Comparison
print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)
print(f"Linear Model R2 Score: {r2:.4f}")
print(f"Polynomial Model R2 Score: {r2_poly:.4f}")
print(f"Linear Model RMSE: ${rmse:,.2f}")
print(f"Polynomial Model RMSE: ${rmse_poly:,.2f}")

if r2_poly > r2:
    print("\nPolynomial Regression performs better (higher R2 score)")
else:
    print("\nLinear Regression performs better (higher R2 score)")