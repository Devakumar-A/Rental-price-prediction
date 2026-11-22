import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset from the 'data' folder
df = pd.read_csv("data/Housing.csv")

# Encode categorical values
df = pd.get_dummies(df, drop_first=True)

# Features & Target
X = df.drop("price", axis=1)
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the columns to get the exact order and names
print("Columns from the trained model (copy this list):")
print(X_train.columns.tolist())

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")