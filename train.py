# train.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Loading Iris dataset")

# 1. Load data
X, y = load_iris(return_X_y=True)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model")

# 3. Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Save model
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.pkl")

#Finally
print("Model training completed and saved")
