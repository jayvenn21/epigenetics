import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
iq_data_path = '/Users/jayanth/Desktop/Personal Projects/Nutrition/archive(8)/avgIQpercountry.csv'
iq_data = pd.read_csv(iq_data_path)

# Display the first few rows of the data
print(iq_data.head())

# Check for missing values
print(iq_data.isna().sum())

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

# Scatter plot of Average IQ vs Population
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iq_data_clean, x='Population - 2023', y='Average IQ', hue='IQ Quartile', palette='viridis')
plt.title('Average IQ vs Population')
plt.xlabel('Population (2023)')
plt.ylabel('Average IQ')
plt.xscale('log')  # Optional: Use log scale if population range is large
plt.legend(title='IQ Quartile')
plt.show()

# Scatter plot of Average IQ vs HDI
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iq_data_clean, x='HDI (2021)', y='Average IQ', hue='IQ Quartile', palette='viridis')
plt.title('Average IQ vs Human Development Index (HDI)')
plt.xlabel('Human Development Index (HDI) (2021)')
plt.ylabel('Average IQ')
plt.legend(title='IQ Quartile')
plt.show()
