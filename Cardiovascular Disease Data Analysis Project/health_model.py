import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load your dataset
df = pd.read_csv("cleaned_health_data.csv")

# Step 1: Define target and features
y = df["Heart_Disease"]
X = df.drop(columns=["Heart_Disease"])  # Keep General_Health this time

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = model.predict(X_test)
print("✅ Model Evaluation Complete")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5: Save the model
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_features.pkl")



