import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load CSV Data
data = pd.read_csv("DWLR_Dataset_2023.csv")

# Standardize column names: lowercase + underscores
data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]

# (Optional) Check column names
# print(data.columns.tolist())
# Expected now:
# ['water_level_m', 'temperature_c', 'rainfall_mm', 'ph', 'dissolved_oxygen_mg_l', ...maybe date...]

# Rename to cleaner names for convenience
data.rename(columns={
    'water_level_m': 'water_level',
    'temperature_c': 'temperature',
    'rainfall_mm': 'rainfall',
    'dissolved_oxygen_mg_l': 'oxygen'   # pH stays as 'ph' already
}, inplace=True)

# Step 2: Data Cleaning
# Use the NEW names (all lowercase)
data = data.dropna(subset=['water_level', 'rainfall', 'temperature', 'ph', 'oxygen'])
data = data.reset_index(drop=True)

# Step 3: Feature Selection
features = ['rainfall', 'temperature', 'ph', 'oxygen']
X = data[features]
y = data['water_level']

# Step 4: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: PCA
pca = PCA(n_components=4)
pca.fit(X_scaled)

# Step 6: Display PCA components
pca_components = pd.DataFrame(
    pca.components_,
    columns=features,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)

print("\n=== PCA Component Loadings ===")
print(pca_components.round(3))

# Step 7: Explained variance
explained_var = pca.explained_variance_ratio_
print("\n=== Explained Variance by Each PC ===")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var*100:.2f}%")

# Step 8: Determine factor importance (from PC1)
loading_scores = np.abs(pca_components.loc['PC1'])
sorted_loadings = loading_scores.sort_values(ascending=False)
print("\n=== Factors Most Influencing Water Level (from PC1) ===")
print(sorted_loadings)

# Step 9: Visualization

# (a) Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(data[['water_level'] + features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Water Level and Other Factors")
plt.show()

# (b) PCA loadings bar chart
plt.figure(figsize=(8, 5))
sorted_loadings.plot(kind='bar', color='teal')
plt.title("Feature Importance (PCA - PC1 Loadings)")
plt.ylabel("Absolute Loading Value")
plt.grid(axis='y')
plt.show()

# (c) Scree plot
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(explained_var) + 1), explained_var * 100, marker='o')
plt.title("Scree Plot (Explained Variance)")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.grid(True)
plt.show()
