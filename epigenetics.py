import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data
iq_data_path = '/Users/jayanth/Desktop/Personal Projects/Nutrition/archive(8)/avgIQpercountry.csv'
nutri_path = '/Users/jayanth/Desktop/Personal Projects/Nutrition/archive(10)/Protein_Supply_Quantity_Data.csv'
risk_factors_path = '/Users/jayanth/Desktop/Personal Projects/Nutrition/number-of-deaths-by-risk-factor.csv'
height_data_path = '/Users/jayanth/Desktop/Personal Projects/Nutrition/annual-change-in-average-male-height.csv'

iq_data = pd.read_csv(iq_data_path)
nutri_data = pd.read_csv(nutri_path)
risk_factors_data = pd.read_csv(risk_factors_path)
height_data = pd.read_csv(height_data_path)

# Display the first few rows of the data
print(iq_data.head())
print(nutri_data.head())
print(risk_factors_data.head())
print(height_data.head())

# Check for missing values in iq_data and nutri_data
print("Missing values in iq_data per column:")
print(iq_data.isna().sum())

print("Missing values in nutri_data per column:")
print(nutri_data.isna().sum())

print("Missing values in risk_factors_data per column:")
print(risk_factors_data.isna().sum())

# Check for missing values in the height data
print("Missing values in height_data per column:")
print(height_data.isna().sum())

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

# Display r and r^2 on the plot
plt.text(0.95, 0.05, f'r: {correlation_coefficient:.2f}\nr^2: {r_squared:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

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

# Display r and r^2 on the plot
plt.text(0.95, 0.05, f'r: {correlation_coefficient:.2f}\nr^2: {r_squared:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

plt.legend(title='IQ Quartile')
plt.show()

# Scatter plot of Average IQ vs GNI - 2021
plt.figure(figsize=(12, 8))
sns.scatterplot(data=iq_data_clean, x='GNI - 2021', y='Average IQ', hue='IQ Quartile', palette='viridis')
plt.title('Average IQ vs GNI - 2021')
plt.xlabel('GNI (2021)')
plt.ylabel('Average IQ')
plt.xscale('log')  # Optional: Use log scale if GNI range is large

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

# Display r and r^2 on the plot
plt.text(0.95, 0.05, f'r: {correlation_coefficient:.2f}\nr^2: {r_squared:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

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

# Fit linear model
X = iq_data_clean[['Mean years of schooling - 2021']]
y = iq_data_clean['Average IQ']
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
correlation_coefficient = np.sqrt(r_squared)
print(f"Correlation coefficient (r) for IQ vs Mean Years of Schooling: {correlation_coefficient}")
print(f"Coefficient of determination (r^2) for IQ vs Mean Years of Schooling: {r_squared}")

# Display r and r^2 on the plot
plt.text(0.95, 0.05, f'r: {correlation_coefficient:.2f}\nr^2: {r_squared:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

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

# Display r and r^2 on the plot
plt.text(0.95, 0.05, f'r: {correlation_coefficient:.2f}\nr^2: {r_squared:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

plt.show()

# Calculate ratios for obesity/malnourished
nutri_data_clean['Obesity_Malnourished_Ratio'] = nutri_data_clean['Obesity'] / nutri_data_clean['Undernourished']

# Create a new variable combining meat, eggs, and milk
nutri_data_clean['Combined_Protein'] = nutri_data_clean['Meat'] + nutri_data_clean['Eggs'] + nutri_data_clean['Milk - Excluding Butter']

# Scatter plot of Milk - Excluding Butter vs. Obesity_Malnourished_Ratio
plt.figure(figsize=(12, 8))
sns.scatterplot(data=nutri_data_clean, x='Milk - Excluding Butter', y='Obesity_Malnourished_Ratio', palette='viridis')
plt.title('Milk - Excluding Butter vs. Obesity/Malnourished Ratio')
plt.xlabel('Milk - Excluding Butter')
plt.ylabel('Obesity/Malnourished Ratio')
plt.xscale('log')  # Optional: Use log scale if range of milk consumption is large

# Label each point with the country name
for i, row in nutri_data_clean.iterrows():
    plt.text(row['Milk - Excluding Butter'], row['Obesity_Malnourished_Ratio'], row['Country'], fontsize=8, alpha=0.75)

# Fit linear model
X = nutri_data_clean[['Milk - Excluding Butter']]
y = nutri_data_clean['Obesity_Malnourished_Ratio']
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
correlation_coefficient = model.coef_[0]
print(f"Coefficient (slope) for Milk - Excluding Butter vs. Obesity/Malnourished Ratio: {correlation_coefficient}")
print(f"Coefficient of determination (r^2) for Milk - Excluding Butter vs. Obesity/Malnourished Ratio: {r_squared}")

# Display r and r^2 on the plot
plt.text(0.95, 0.05, f'r: {correlation_coefficient:.2f}\nr^2: {r_squared:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

plt.show()

# Scatter plot of Combined_Protein vs. Obesity_Malnourished_Ratio
plt.figure(figsize=(12, 8))
sns.scatterplot(data=nutri_data_clean, x='Combined_Protein', y='Obesity_Malnourished_Ratio', palette='viridis')
plt.title('Combined Protein vs. Obesity/Malnourished Ratio')
plt.xlabel('Combined Protein (Meat + Eggs + Milk)')
plt.ylabel('Obesity/Malnourished Ratio')

# Label each point with the country name
for i, row in nutri_data_clean.iterrows():
    plt.text(row['Combined_Protein'], row['Obesity_Malnourished_Ratio'], row['Country'], fontsize=8, alpha=0.75)

# Fit linear model
X = nutri_data_clean[['Combined_Protein']]
y = nutri_data_clean['Obesity_Malnourished_Ratio']
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
correlation_coefficient = model.coef_[0]
print(f"Coefficient (slope) for Combined Protein vs. Obesity/Malnourished Ratio: {correlation_coefficient}")
print(f"Coefficient of determination (r^2) for Combined Protein vs. Obesity/Malnourished Ratio: {r_squared}")

# Display r and r^2 on the plot
plt.text(0.95, 0.05, f'r: {correlation_coefficient:.2f}\nr^2: {r_squared:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

plt.show()


# Extract data for India and specific columns of interest
india_data = risk_factors_data[(risk_factors_data['Entity'] == 'India') & (risk_factors_data['Year'] >= 1990) & (risk_factors_data['Year'] <= 2017)]
columns_of_interest = ['Year', 'Iron deficiency', 'Low bone mineral density']
india_data = india_data[columns_of_interest]

# Plotting the bar graph for Iron deficiency
plt.figure(figsize=(12, 8))
sns.barplot(data=india_data, x='Year', y='Iron deficiency', color='red')
plt.title('Iron Deficiency in India (1990 - 2017)')
plt.xlabel('Year')
plt.ylabel('Number of Deaths due to Iron Deficiency')
plt.show()

# Scatter plot for the correlation between Iron deficiency and Low bone mineral density
plt.figure(figsize=(12, 8))
sns.scatterplot(data=india_data, x='Iron deficiency', y='Low bone mineral density')
plt.title('Correlation between Iron Deficiency and Low Bone Mineral Density in India')
plt.xlabel('Number of Deaths due to Iron Deficiency')
plt.ylabel('Number of Deaths due to Low Bone Mineral Density')

# Fit linear model
X = india_data[['Iron deficiency']]
y = india_data['Low bone mineral density']
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
correlation_coefficient = np.sqrt(r_squared)
print(f"Correlation coefficient (r) for Iron Deficiency vs Low Bone Mineral Density: {correlation_coefficient}")
print(f"Coefficient of determination (r^2) for Iron Deficiency vs Low Bone Mineral Density: {r_squared}")

# Display r and r^2 on the plot
plt.text(0.95, 0.05, f'r: {correlation_coefficient:.2f}\nr^2: {r_squared:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

plt.show()


# Analysis on Mean Male and Female Height

# Bar plot for Mean Male Height (cm) in 1996
plt.figure(figsize=(14, 10))
height_1996 = height_data[height_data['Year'] == 1996]
height_1996_sorted_male = height_1996.sort_values(by='Mean male height (cm)')
sns.barplot(data=height_1996_sorted_male, x='Entity', y='Mean male height (cm)', palette='viridis')
plt.title('Mean Male Height (cm) in 1996')
plt.xlabel('Country')
plt.ylabel('Mean Male Height (cm)')
plt.xticks(rotation=90, ha='right')
plt.show()

# Bar plot for Mean Female Height (cm) in 1996
plt.figure(figsize=(14, 10))
height_1996_sorted_female = height_1996.sort_values(by='Mean female height (cm)')
sns.barplot(data=height_1996_sorted_female, x='Entity', y='Mean female height (cm)', palette='viridis')
plt.title('Mean Female Height (cm) in 1996')
plt.xlabel('Country')
plt.ylabel('Mean Female Height (cm)')
plt.xticks(rotation=90, ha='right')
plt.show()

# Ensure each country has enough space in the text
plt.yticks(rotation=0, ha='right')
plt.show()

# Filter the height data for India from 1896 to 1996
height_india = height_data[(height_data['Entity'] == 'India') & (height_data['Year'] >= 1896) & (height_data['Year'] <= 1996)]

# Bar plot for Mean Male Height (cm) in India from 1896 to 1996
plt.figure(figsize=(12, 8))
sns.barplot(data=height_india, x='Year', y='Mean male height (cm)', palette='viridis')
plt.title('Mean Male Height (cm) in India (1896-1996)')
plt.xlabel('Year')
plt.ylabel('Mean Male Height (cm)')
plt.xticks(rotation=90)
plt.show()

# Bar plot for Mean Female Height (cm) in India from 1896 to 1996
plt.figure(figsize=(12, 8))
sns.barplot(data=height_india, x='Year', y='Mean female height (cm)', palette='viridis')
plt.title('Mean Female Height (cm) in India (1896-1996)')
plt.xlabel('Year')
plt.ylabel('Mean Female Height (cm)')
plt.xticks(rotation=90)
plt.show()