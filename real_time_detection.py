# real_time_detection.py

import cv2
import joblib
from feature_extraction import extract_features_from_frame

# Load the trained model
model = joblib.load('models/model.pkl')

def predict_distance(frame):
    """Predict the distance based on the extracted features from the frame."""
    features = extract_features_from_frame(frame)
    return model.predict([features])

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    distance = predict_distance(frame)
    cv2.putText(frame, f'Distance: {distance[0]:.2f} m', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Distance Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
