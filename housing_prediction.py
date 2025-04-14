
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap

df = pd.read_csv("/Users/rushabarram/Desktop/big_data_projects/Housing.csv")

# Basic cleaning (convert categorical variables to lowercase)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.lower()

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Split into features and target
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Feature Importance
importances = model.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# SHAP for Explainability
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test)

# SHAP dependence plot example
shap.dependence_plot("area", shap_values.values, X_test)

# Save model (optional)
# import joblib
# joblib.dump(model, "housing_price_model.pkl")
