import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Drop rows with missing values in critical columns for the analysis
iq_data_clean = iq_data.dropna(subset=['Average IQ', 'Population - 2023', 'HDI (2021)'])

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
