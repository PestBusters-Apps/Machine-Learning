from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import io

# Initialize Flask app
app = Flask(__name__)

# Set model path and verify existence
MODEL_PATH = "yolo11n.pt"  # Replace with your model filename if needed
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load YOLO model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Define a function for image preprocessing
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to the model's expected size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define route to handle image uploads
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read image
    file = request.files['image']
    image_bytes = file.read()

    try:
        # Preprocess the image
        input_tensor = preprocess_image(image_bytes)

        # Perform inference
        with torch.no_grad():
            predictions = model(input_tensor.to(device))

        # Process predictions (customize this as per your model output format)
        results = []
        for pred in predictions:
            for *box, conf, cls in pred:
                results.append({
                    'box': [round(x.item(), 2) for x in box],
                    'confidence': round(conf.item(), 2),
                    'class': int(cls.item())
                })

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)