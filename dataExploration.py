"""
Data Exploration Script for Movie Dataset Analysis
This script performs initial data exploration and preprocessing of the movie metadata dataset.
It includes data cleaning, feature engineering, and basic statistical analysis.
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Load the movie metadata dataset
df = pd.read_csv('/Users/user/OneDrive/바탕 화면/데과 텀프/movies_metadata.csv', low_memory=False)

# Select relevant features for analysis
columns_to_use = [
    'budget', 'genres', 'original_language', 'popularity',
    'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count'
]
# columns_to_use = [
#     'budget', 'genres', 'original_language',
#     'release_date', 'runtime'
# ]
df = df[columns_to_use]

# Convert string columns to numeric type for analysis
numeric_columns = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count']
# numeric_columns = ['budget', 'runtime']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create a binary success label based on revenue/budget ratio
# A movie is considered successful if it makes at least 2x its budget
df['success'] = (df['revenue'] / df['budget']) >= 2
df['success'] = df['success'].map({True: 'Success', False: 'Fail'})

# Remove invalid entries and missing values
df_cleaned = df[(df['budget'] > 0) & (df['revenue'] > 0)].dropna(subset=['budget', 'revenue'])

# Scale budget and revenue features for better comparison
scaler = StandardScaler()
df_cleaned[['budget_scaled', 'revenue_scaled']] = scaler.fit_transform(df_cleaned[['budget', 'revenue']])

# Generate and print statistical summaries
print("\n[1] Numerical Feature Description")
print(df_cleaned[numeric_columns].describe())

# Check for missing values in the dataset
print("\n[2] Missing Values per Column")
print(df[columns_to_use].isnull().sum())

# Analyze categorical features
print("\n[3] Top 5 Original Languages")
print(df['original_language'].value_counts().head(5))

print("\n[4] Sample Genres Data")
print(df['genres'].dropna().head(5))

# Analyze success distribution
print("\n[5] Success vs Fail Counts")
print(df_cleaned['success'].value_counts())

# Display scaled features
print("\n[6] Scaled Budget and Revenue")
print(df_cleaned[['budget', 'revenue']].head())
print(df_cleaned[['budget_scaled', 'revenue_scaled']].head())

# Create and save correlation heatmap
correlation = df_cleaned[numeric_columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig('./correlation_heatmap.png')
plt.show()

# Process and encode genre information
df['genres'] = df['genres'].fillna('[]')  # Replace missing values with empty list
df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
df['main_genre'] = df['genres'].apply(lambda x: x[0]['name'] if len(x) > 0 else 'None')  # Extract main genre
df_encoded_genres = pd.get_dummies(df['main_genre'], prefix='genre')  # One-Hot Encoding for genres

# Process and encode language information
df['original_language'] = df['original_language'].fillna('unknown')  # Replace missing values with 'unknown'
df_encoded_languages = pd.get_dummies(df['original_language'], prefix='lang')  # One-Hot Encoding for languages

