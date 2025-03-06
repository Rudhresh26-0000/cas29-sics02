import cv2
import numpy as np
import tensorflow as tf

# Load the MobileNet model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Preprocess the frame
    frame = cv2.resize(frame, (224, 224))
    frame = tf.keras.applications.mobilenet.preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)

    # Make predictions
    predictions = model.predict(frame)

    # Get the top prediction
    predicted_class = tf.keras.applications.mobilenet.decode_predictions(predictions, top=1)[0][0]
    
    # Display the prediction on the frame
    cv2.putText(frame, "Predicted: {}".format(predicted_class[1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Smart Agriculture Monitoring', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()