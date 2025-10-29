
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

# --- 1. Configuration and Path Setup ---
# Use relative paths for portability
SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_DIR, '../../data/amazon.csv')
MODEL_OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../api/model')

# Load the dataset
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# --- 2. Data Cleaning and Preprocessing ---
print("Cleaning and preprocessing data...")
df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)
df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float) # Keep this here for feature engineering
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

# --- 3. Feature Engineering ---
print("Performing feature engineering...")
# Split the category string into multiple hierarchical features
category_levels = df['category'].str.split('|', expand=True)

# Assign up to 5 levels of categories. Use 'Unknown' for levels that don't exist in a given row.
df['main_category_name'] = category_levels[0] # Keep original names for grouping
df['main_category'] = category_levels[0]
df['sub_category_1'] = category_levels[1].fillna('Unknown')
df['sub_category_2'] = category_levels[2].fillna('Unknown')
df['sub_category_3'] = category_levels[3].fillna('Unknown')
df['sub_category_4'] = category_levels[4].fillna('Unknown') # Assuming max 5 levels based on example
df['sub_category_5'] = category_levels[5].fillna('Unknown')
df['sub_category_6'] = category_levels[6].fillna('Unknown')

# Fill any missing sub-categories that might result from the split
# The fillna for main_category and sub_category_1 is now handled by the above assignment

# Label encode the new categorical features
categorical_features_to_encode = ['main_category', 'sub_category_1', 'sub_category_2', 'sub_category_3', 'sub_category_4', 'sub_category_5', 'sub_category_6']
for col in categorical_features_to_encode:
    df[col] = df[col].astype('category').cat.codes

# --- 3a. Intelligent Imputation of Missing Values (after Feature Engineering) ---
print("Intelligently imputing missing 'rating' and 'rating_count' values...")
# Calculate median rating and rating_count per main_category
median_rating_by_category = df.groupby('main_category_name')['rating'].median()
median_rating_count_by_category = df.groupby('main_category_name')['rating_count'].median()

# Impute missing 'rating' values with the median rating of their respective main_category
# If a main_category itself has no ratings, fall back to the overall median rating
df['rating'] = df.apply(
    lambda row: row['rating'] if pd.notna(row['rating']) else \
                median_rating_by_category.get(row['main_category_name'], df['rating'].median()),
    axis=1
)

# Impute missing 'rating_count' values similarly
df['rating_count'] = df.apply(
    lambda row: row['rating_count'] if pd.notna(row['rating_count']) else \
                median_rating_count_by_category.get(row['main_category_name'], df['rating_count'].median()),
    axis=1
)

# Drop the temporary 'main_category_name' column
df.drop(columns=['main_category_name'], inplace=True)


# --- 4. Feature Selection and Data Splitting ---
# Use the new, more granular category features
# Add the newly created sub-category features
features = categorical_features_to_encode + ['actual_price', 'rating', 'rating_count']
target = 'discount_percentage'

print(f"Using features: {features}")
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Hyperparameter Tuning with RandomizedSearchCV ---
print("Starting hyperparameter tuning with RandomizedSearchCV for XGBoost...")

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
}

# Initialize the model
xgboost = xgb.XGBRegressor(random_state=42, objective='reg:absoluteerror')

# Split the training data to create a validation set for early stopping
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# This is the correct way to use early stopping with scikit-learn's CV tools.
random_search = RandomizedSearchCV(
    estimator=xgboost,
    param_distributions=param_grid,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=3,       # 3-fold cross-validation
    scoring='neg_mean_absolute_error',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Fitting the model...")
# Use a callback for early stopping
random_search.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)]) # , callbacks=[xgb.callback.EarlyStopping(rounds=10)], verbose=False)

print(f"Best parameters found: {random_search.best_params_}")
best_model = random_search.best_estimator_

# --- 6. Evaluate the Best Model ---
print("\nEvaluating the best model found...")
y_pred = best_model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# --- 6a. Feature Importance ---
print("\nFeature Importances:")
feature_importances = pd.DataFrame({'feature': features, 'importance': best_model.feature_importances_})
feature_importances = feature_importances.sort_values('importance', ascending=False)
print(feature_importances)

# --- 7. Save the Model and Columns ---
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODEL_OUTPUT_DIR, 'discount_predictor.pkl'))
joblib.dump(features, os.path.join(MODEL_OUTPUT_DIR, 'model_columns.pkl'))

print(f"\nModel and columns saved successfully to {MODEL_OUTPUT_DIR}")
