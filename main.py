import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Load the dataset
data_path = 'q111.csv'
df = pd.read_csv(data_path)

# Preprocessing
# Drop the Response ID as it is not a feature
df = df.drop(columns=['Response ID'])

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Define features (X) and target (y)
X = df.drop(columns=['Which work type do you prefer the most?'])
y = df['Which work type do you prefer the most?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False)

# Train each model
rf_model.fit(X_train_res, y_train_res)
gb_model.fit(X_train_res, y_train_res)
xgb_model.fit(X_train_res, y_train_res)

# Make predictions using each model
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the models
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"Accuracy (Random Forest): {accuracy_rf * 100:.2f}%")
print("\nClassification Report (Random Forest):\n")
print(classification_report(y_test, y_pred_rf))

print(f"Accuracy (Gradient Boosting): {accuracy_gb * 100:.2f}%")
print("\nClassification Report (Gradient Boosting):\n")
print(classification_report(y_test, y_pred_gb))

print(f"Accuracy (XGBoost): {accuracy_xgb * 100:.2f}%")
print("\nClassification Report (XGBoost):\n")
print(classification_report(y_test, y_pred_xgb))

# Combine predictions (Majority voting manually)
y_pred_combined = []
for i in range(len(y_pred_xgb)):
    # Majority vote: take the most frequent class label among the models
    vote = [y_pred_xgb[i], y_pred_rf[i], y_pred_gb[i]]
    y_pred_combined.append(max(set(vote), key=vote.count))

# Evaluate the combined model
accuracy_combined = accuracy_score(y_test, y_pred_combined)
print(f"Accuracy (Combined Model): {accuracy_combined * 100:.2f}%")
print("\nClassification Report (Combined Model):\n")
print(classification_report(y_test, y_pred_combined))

# Feature Importance from the best model (Random Forest in this case)
feature_importances = pd.DataFrame(
    rf_model.feature_importances_, index=X.columns, columns=['Importance']
).sort_values(by='Importance', ascending=False)

print("\nFeature Importances (Best Random Forest Model):\n")
print(feature_importances)





