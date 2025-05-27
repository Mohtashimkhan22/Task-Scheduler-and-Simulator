import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the dataset
data = pd.read_csv('Crop_recommendation.csv', encoding='utf-8')
x = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels

# Train the model
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open('crop_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the saved model
model = pickle.load(open('crop_model.pkl', 'rb'))

# Function for making predictions
def predict_crop(features):
    feature_names = x.columns  
    features_df = pd.DataFrame([features], columns=feature_names)
    
    return model.predict(features_df)[0]

# Example usage
if __name__ == "__main__":
    example_features = [300, 410, 200, 10, 30, 10, 5]  # Example input
    predicted_crop = predict_crop(example_features)
    print("Predicted Crop:", predicted_crop)
