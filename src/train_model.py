import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Separate features and label
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_resampled, y_resampled)

# Predict
y_pred = model.predict(X_test)

# Show performance
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("fraud_model.pkl", "wb"))

print("Model saved as fraud_model.pkl")
