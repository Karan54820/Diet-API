import os
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# Load dataset
file_path = "food_dataset.csv"
data = pd.read_csv(file_path)

# Feature selection
start_nutrients_idx = 2
end_nutrients_idx = 33
X = data.iloc[:, start_nutrients_idx:end_nutrients_idx + 1]

# Disease mapping
disease_map = {
    'anemia': 'suitable_for_anemia',
    'diabetes_type1': 'suitable_for_diabetes_type1',
    'diabetes_type2': 'suitable_for_diabetes_type2',
    'hypertension': 'suitable_for_hypertension',
    'osteoporosis': 'suitable_for_osteoporosis',
    'obesity': 'suitable_for_obesity'
}

# Train and store models
if not os.path.exists("models"):
    os.makedirs("models")

for disease, target_col in disease_map.items():
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    knn = KNeighborsClassifier()
    param_grid = {"n_neighbors": range(1, 50)}
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train_scaled, y_train)

    best_k = grid_search.best_params_["n_neighbors"]
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)

    # Save model & scaler
    with open(f"models/{disease}_model.pkl", "wb") as f:
        pickle.dump(best_knn, f)

    with open(f"models/{disease}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
