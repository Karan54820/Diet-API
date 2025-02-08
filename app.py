from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

app = Flask(__name__)

# Disease mapping
disease_map = {
    'anemia': 'suitable_for_anemia',
    'diabetes_type1': 'suitable_for_diabetes_type1',
    'diabetes_type2': 'suitable_for_diabetes_type2',
    'hypertension': 'suitable_for_hypertension',
    'osteoporosis': 'suitable_for_osteoporosis',
    'obesity': 'suitable_for_obesity'
}

# Load the dataset
file_path = "food_dataset.csv"
data = pd.read_csv(file_path)

# Feature selection
start_nutrients_idx = 2
end_nutrients_idx = 33
X = data.iloc[:, start_nutrients_idx:end_nutrients_idx + 1]

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts food recommendations based on disease."""
    data = request.json
    user_disease = data.get("disease", "").lower()

    if user_disease not in disease_map:
        return jsonify({"error": "Invalid disease"}), 400

    target_col = disease_map[user_disease]

    # Load pre-trained model & scaler
    model_path = f"models/{user_disease}_model.pkl"
    scaler_path = f"models/{user_disease}_scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return jsonify({"error": "Model or scaler not found"}), 500

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Get recommended nutrients
    suitable_recipes = X[data[target_col] == 1]
    unsuitable_recipes = X[data[target_col] == 0]
    suitable_nutrients = suitable_recipes.mean()
    unsuitable_nutrients = unsuitable_recipes.mean()
    nutrient_importance = (suitable_nutrients - unsuitable_nutrients) / unsuitable_nutrients
    recommended_nutrients = nutrient_importance.nlargest(5).index.tolist()

    # Recommend top 5 recipes
    suitable_recipes = data[data[target_col] == 1]
    nutrient_scores = suitable_recipes[recommended_nutrients].mean(axis=1)
    top_recipes = suitable_recipes.loc[nutrient_scores.nlargest(5).index, "recipe_name"].tolist()

    return jsonify({
        "disease": user_disease,
        "recommended_nutrients": recommended_nutrients,
        "top_recipes": top_recipes
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)
