import serial
import time
import numpy as np
import tensorflow as tf
import cv2  # OpenCV for webcam and display

# Initialize serial communication
ser = serial.Serial('COM3', 9600)  # Update 'COM3' to your Arduino's port
time.sleep(2)  # Wait for the connection to establish

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")  # Replace with your model's path
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_frame(frame):
    """
    Preprocess the input frame to match the model's expected input size and type.
    Adjusted for UINT8 input.
    """
    input_shape = input_details[0]['shape']  # Model's expected input shape
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[2]))  # Resize to model's input size
    
    if input_details[0]['dtype'] == np.uint8:  # For UINT8 models
        input_data = np.array(frame_resized, dtype=np.uint8)  # Ensure UINT8 type
    else:  # For FLOAT32 models
        input_data = np.array(frame_resized, dtype=np.float32) / 255.0  # Normalize to 0-1 range
    
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    return input_data

def classify_frame(frame):
    """
    Classify the frame using the TFLite model.
    Returns the predicted class index.
    """
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)  # Return the class index

def send_command(command):
    """
    Send the command to the Arduino via serial.
    """
    ser.write((command + '\n').encode())
    print(f"Sent: {command}")

# Initialize webcam
cap = cv2.VideoCapture(1)  # Change '1' to the desired webcam index

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Perform classification
        prediction = classify_frame(frame)

        # Determine action based on prediction
        if prediction == 1:  # Recyclable
            label = "Recyclable"
            send_command("clockwise")
        elif prediction == 0:  # Non-recyclable
            label = "Non-recyclable"
            send_command("anticlockwise")
        else:  # None
            label = "None (No action)"

        # Display the label on the frame
        cv2.putText(frame, f"Class: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Classification", frame)

        # Delay to allow motor rotation to complete
        if prediction == 0 or prediction == 1:  # If action was taken
            print("Waiting for motor rotation to complete...")
            time.sleep(2)  # Adjust delay as needed for motor rotation

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program.")
            break

except KeyboardInterrupt:
    print("Program interrupted.")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("Webcam and serial connection closed.")
