import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# ----------------- pCloud API Integration -----------------
API_TOKEN = "your_pcloud_api_token"

def get_user_info():
    # Replace with your actual pCloud API URL
    url = "https://api.pcloud.com/userinfo"
    params = {"auth": API_TOKEN}
    response = requests.get(url, params=params)
    return response.json()

# Fetch pCloud user info
user_info = get_user_info()
print("User Info:", user_info)

# ----------------- Crop Prediction Model -----------------
# Sample dataset
data = {
    'Temperature': [30, 25, 20, 35, 40, 22, 28, 24, 27, 32],
    'Humidity': [80, 65, 60, 90, 75, 55, 70, 50, 85, 88],
    'pH': [6.5, 7.0, 5.5, 6.8, 7.2, 5.8, 6.0, 6.3, 7.5, 6.7],
    'Rainfall': [200, 150, 100, 250, 180, 120, 160, 130, 220, 210],
    'Crop': ['Rice', 'Wheat', 'Barley', 'Sugarcane', 'Maize', 'Lentil', 'Millet', 'Soybean', 'Cotton', 'Banana']
}

df = pd.DataFrame(data)
df['Crop'] = df['Crop'].astype('category').cat.codes  # Encode target variable

X = df.drop('Crop', axis=1)
y = df['Crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy:.2f}")

def predict_crop(temp, humidity, ph, rainfall):
    input_data = np.array([[temp, humidity, ph, rainfall]])
    crop_index = rf_model.predict(input_data)[0]
    crop_name = df['Crop'].astype('category').cat.categories[crop_index]
    return crop_name

# ----------------- MobileNetV2 for Object & Leaf Health Detection -----------------
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')

def predict_leaf_health(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_input = preprocess_input(np.expand_dims(frame_rgb, axis=0))
    
    predictions = mobilenet_model.predict(frame_input)
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]
    
    label = decoded_predictions[1]
    confidence = decoded_predictions[2]
    
    return label, confidence

# ----------------- Webcam Video Processing -----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Object & Leaf Health Prediction
    label, confidence = predict_leaf_health(frame)

    # Display predictions
    cv2.putText(frame, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence*100:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Example: Predict crop based on input conditions
    predicted_crop = predict_crop(29, 70, 6.5, 180)
    print(f"Predicted Crop: {predicted_crop}")

    # Display frame
    cv2.imshow('Smart Agriculture Monitoring - Object & Leaf Health Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()