import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Load the data
iq_data_path = '/Users/jayanth/Desktop/Personal Projects/Nutrition/archive(8)/avgIQpercountry.csv'
nutri_path = '/Users/jayanth/Desktop/Personal Projects/Nutrition/archive(10)/Protein_Supply_Quantity_Data.csv'
iq_data = pd.read_csv(iq_data_path)
nutri_data = pd.read_csv(nutri_path)

# Display the first few rows of the data
print(iq_data.head())
print(nutri_data.head())

# Check for missing values in iq_data and nutri_data
print("Missing values in iq_data per column:")
print(iq_data.isna().sum())

print("Missing values in nutri_data per column:")
print(nutri_data.isna().sum())

# Ensure 'Population - 2023' is numeric
iq_data['Population - 2023'] = pd.to_numeric(iq_data['Population - 2023'], errors='coerce')

print(iq_data.columns)

# Ensure 'Population ' is numeric
nutri_data['Population'] = pd.to_numeric(nutri_data['Population'], errors='coerce')

print(nutri_data.columns)


# Drop rows with missing values in critical columns for the analysis
iq_data_clean = iq_data.dropna(subset=['Average IQ', 'Population - 2023', 'HDI (2021)', 'Mean years of schooling - 2021', 'GNI - 2021'])
print(iq_data_clean.columns)

nutri_data_clean = nutri_data.dropna(subset=['Cereals - Excluding Beer', 'Eggs', 'Fish, Seafood', 'Meat', 'Milk - Excluding Butter', 'Offals', 'Vegetal Products', 'Obesity', 'Undernourished'])
print(nutri_data_clean.columns)


# Calculate quartiles
quartiles = iq_data_clean['Average IQ'].quantile([0.25, 0.5, 0.75])
print(quartiles)

# Function to determine the quartile of a given value
def assign_quartile(value, quartiles):
    if value <= quartiles[0.25]:
        return '1st Quartile'
    elif value <= quartiles[0.5]:
        return '2nd Quartile'
    elif value <= quartiles[0.75]:
        return '3rd Quartile'
    else:
        return '4th Quartile'

# Assign quartiles to each country
iq_data_clean['IQ Quartile'] = iq_data_clean['Average IQ'].apply(assign_quartile, quartiles=quartiles)

# Convert population to millions for plotting
iq_data_clean['Population (Millions)'] = iq_data_clean['Population - 2023'] / 1e6

# Scatter plot of Average IQ vs Population (in millions)
plt.figure(figsize=(12, 8))
sns.scatterplot(data=iq_data_clean, x='Population (Millions)', y='Average IQ', hue='IQ Quartile', palette='viridis')
plt.title('Average IQ vs Population')
plt.xlabel('Population (Millions) (2023)')
plt.ylabel('Average IQ')
plt.xscale('log')  # Optional: Use log scale if population range is large

# Label each point with the country name
for i, row in iq_data_clean.iterrows():
    plt.text(row['Population (Millions)'], row['Average IQ'], row['Country'], fontsize=8, alpha=0.75)

# Fit linear model
X = iq_data_clean[['Population (Millions)']]
y = iq_data_clean['Average IQ']
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
correlation_coefficient = np.sqrt(r_squared)
print(f"Correlation coefficient (r) for IQ vs Population: {correlation_coefficient}")
print(f"Coefficient of determination (r^2) for IQ vs Population: {r_squared}")

plt.legend(title='IQ Quartile')
plt.show()

# Scatter plot of Average IQ vs HDI
plt.figure(figsize=(12, 8))
sns.scatterplot(data=iq_data_clean, x='HDI (2021)', y='Average IQ', hue='IQ Quartile', palette='viridis')
plt.title('Average IQ vs Human Development Index (HDI)')
plt.xlabel('Human Development Index (HDI) (2021)')
plt.ylabel('Average IQ')

# Label each point with the country name
for i, row in iq_data_clean.iterrows():
    plt.text(row['HDI (2021)'], row['Average IQ'], row['Country'], fontsize=8, alpha=0.75)


# Fit linear model
X = iq_data_clean[['HDI (2021)']]
y = iq_data_clean['Average IQ']
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
correlation_coefficient = np.sqrt(r_squared)
print(f"Correlation coefficient (r) for IQ vs HDI: {correlation_coefficient}")
print(f"Coefficient of determination (r^2) for IQ vs HDI: {r_squared}")

plt.legend(title='IQ Quartile')
plt.show()

# Scatter plot of Average IQ vs GNI - 2021
plt.figure(figsize=(12, 8))
sns.scatterplot(data=iq_data_clean, x='GNI - 2021', y='Average IQ', hue='IQ Quartile', palette='viridis')
plt.title('Average IQ vs GNI - 2021')
plt.xlabel('GNI (2021)')
plt.ylabel('Average IQ')
plt.xscale('log')  # Optional: Use log scale if population range is large

# Label each point with the country name
for i, row in iq_data_clean.iterrows():
    plt.text(row['GNI - 2021'], row['Average IQ'], row['Country'], fontsize=8, alpha=0.75)

# Fit linear model
X = iq_data_clean[['GNI - 2021']]
y = iq_data_clean['Average IQ']
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
correlation_coefficient = np.sqrt(r_squared)
print(f"Correlation coefficient (r) for IQ vs GNI: {correlation_coefficient}")
print(f"Coefficient of determination (r^2) for IQ vs GNI: {r_squared}")

plt.legend(title='IQ Quartile')
plt.show()

# Scatter plot of Average IQ vs Mean years of schooling - 2021
plt.figure(figsize=(12, 8))
sns.scatterplot(data=iq_data_clean, x='Mean years of schooling - 2021', y='Average IQ', hue='IQ Quartile', palette='viridis')
plt.title('Average IQ vs Mean Years of Schooling - 2021')
plt.xlabel('Mean Years of Schooling - 2021')
plt.ylabel('Average IQ')
plt.xscale('log')  # Optional: Use log scale if range of schooling years is large

# Label each point with the country name
for i, row in iq_data_clean.iterrows():
    plt.text(row['Mean years of schooling - 2021'], row['Average IQ'], row['Country'], fontsize=8, alpha=0.75)

# Fit quadratic model
X = iq_data_clean[['Mean years of schooling - 2021']]
y = iq_data_clean['Average IQ']

# Using PolynomialFeatures to transform input features
degree = 2  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Create a pipeline with PolynomialFeatures and LinearRegression
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)

# Predict using the model to plot the curve
X_plot = np.linspace(X.min(), X.max(), 100)[:, np.newaxis]
y_plot = model.predict(X_plot.reshape(-1, 1))

# Flatten X_plot and y_plot to 1-dimensional arrays
X_plot = X_plot.flatten()
y_plot = y_plot.flatten()

# Plot the quadratic curve
plt.plot(X_plot, y_plot, color='red', label=f'Quadratic Fit (Degree {degree})')

# Calculate R^2 for the quadratic model
r_squared = model.score(X, y)
correlation_coefficient = np.sqrt(r_squared)
print(f"Correlation coefficient (r) for IQ vs Mean Years of Schooling (Quadratic): {correlation_coefficient}")
print(f"Coefficient of determination (r^2) for IQ vs Mean Years of Schooling (Quadratic): {r_squared}")

plt.legend(title='IQ Quartile')
plt.show()




# Convert '<2.5' to a numeric value, e.g., 2.0
nutri_data_clean['Undernourished'] = nutri_data_clean['Undernourished'].apply(lambda x: 2.0 if x == '<2.5' else float(x))

# Scatter plot of Milk - Excluding Butter vs. Undernourished
plt.figure(figsize=(12, 8))
sns.scatterplot(data=nutri_data_clean, x='Milk - Excluding Butter', y='Undernourished', palette='viridis')
plt.title('Milk - Excluding Butter vs. Undernourished')
plt.xlabel('Milk - Excluding Butter')
plt.ylabel('Undernourished')
plt.xscale('log')  # Optional: Use log scale if population range is large

# Label each point with the country name
for i, row in nutri_data_clean.iterrows():
    plt.text(row['Milk - Excluding Butter'], row['Undernourished'], row['Country'], fontsize=8, alpha=0.75)

# Fit linear model
X = nutri_data_clean[['Milk - Excluding Butter']]
y = nutri_data_clean['Undernourished']
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
correlation_coefficient = model.coef_[0]
print(f"Coefficient (slope) for Milk - Excluding Butter vs. Undernourished: {correlation_coefficient}")
print(f"Coefficient of determination (r^2) for Milk - Excluding Butter vs. Undernourished: {r_squared}")

plt.show()

