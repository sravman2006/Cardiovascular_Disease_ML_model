import joblib
import pandas as pd

# Load the trained model and expected feature list
model = joblib.load("heart_disease_model.pkl")
expected_features = joblib.load("model_features.pkl")

def ask_user():
    print("🩺 Let's assess your cardiovascular risk. Please answer the following:")

    general_health = int(input("General Health (0=Poor, 1=Fair, 2=Good, 3=Very Good, 4=Excellent): "))
    checkup = int(input("Last Checkup (0=Never, 1=5+ years ago, 2=within 5 years, 3=within 2 years, 4=within 1 year): "))
    exercise = int(input("Do you exercise? (0=No, 1=Yes): "))
    age_category = int(input("Age Category (0=18-24, 1=25-29, ..., 11=80+): "))
    sex = int(input("Sex (0=Male, 1=Female): "))
    smoking = int(input("Smoking History (0=No, 1=Yes): "))
    diabetes = int(input("Diabetes (0=No, 1=Yes): "))
    depression = int(input("Depression (0=No, 1=Yes): "))
    alcohol = float(input("Alcohol Consumption (drinks/week): "))
    fruit = float(input("Fruit Consumption (servings/day): "))
    veg = float(input("Green Veg Consumption (servings/day): "))
    fried = float(input("Fried Potato Consumption (servings/week): "))
    height = float(input("Height (cm): "))
    weight = float(input("Weight (kg): "))

    bmi = weight / ((height / 100) ** 2)

    # Create input dictionary
    input_dict = {
        "General_Health": general_health,
        "Checkup": checkup,
        "Exercise": exercise,
        "Age_Category": age_category,
        "Sex": sex,
        "Smoking_History": smoking,
        "Diabetes": diabetes,
        "Depression": depression,
        "Alcohol_Consumption": alcohol,
        "Fruit_Consumption": fruit,
        "Green_Vegetables_Consumption": veg,
        "FriedPotato_Consumption": fried,
        "Height_(cm)": height,
        "Weight_(kg)": weight,
        "BMI": round(bmi, 2),
        "Arthritis": 0,
        "Heart_Disease": 0,
        "Other_Cancer": 0,
        "Skin_Cancer": 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure all expected features are present
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value for missing features

    # Reorder columns to match training
    input_df = input_df[expected_features]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    # Output
    print("\n🧠 Prediction:", "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease")
    print(f"🔢 Confidence: {probability:.2%}")

ask_user()

