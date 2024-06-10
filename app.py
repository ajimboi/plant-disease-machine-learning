import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load_model('HARUMANISDISEASENEW_model.h5')

# Function to preprocess input image
def preprocess_image(image):
    # Resize the image to match model input size
    resized_image = cv2.resize(image, (150, 150))  # Adjust size if needed
    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    # Normalize the image
    normalized_image = rgb_image / 255.0
    return normalized_image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

       

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Save the image file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Read the saved image
        image = cv2.imread(file_path)

        # Preprocess the input image
        processed_image = preprocess_image(image)

        # Reshape the processed image to match the expected input shape of the model
        processed_image = np.expand_dims(processed_image, axis=0)

        # Get predictions from the model
        predictions = model.predict(processed_image)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions[0])

        # Get the probability of the predicted class
        predicted_class_probability = predictions[0][predicted_class_index]
        print(predicted_class_index )
        print(predicted_class_probability)
        

        # Return the predicted class and its associated probability
        return jsonify({'predicted_class': int(predicted_class_index), 'predicted_class_probability': float(predicted_class_probability)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
