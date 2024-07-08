from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
import base64

app = Flask(__name__)

class_names = {
    0: 'red',
    1: 'white',
    2: 'purple',
    3: 'pink',
    4: 'blue',
    5: 'orange',
    6: 'black',
    7: 'silver',
    8: 'brown',
    9: 'green',
    10: 'yellow',
    11: 'grey'
}

num_labels = 12  # Replace this with the actual number of classes

# Define the model architecture
class ColorClassifier(nn.Module):
    def __init__(self):
        super(ColorClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=6, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Load the pre-trained model
model = ColorClassifier()
state_dict = torch.load("color_classifier.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        img = Image.open(file).convert('RGB')
        img_original = img.copy()  # Make a copy for display purposes
        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        # Get the class name from class_names dictionary
        if predicted_class in class_names:
            predicted_color = class_names[predicted_class]
        else:
            predicted_color = 'unknown'

        # Convert PIL image to base64 encoded string for display in HTML
        img_byte_arr = io.BytesIO()
        img_original.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        img_str = 'data:image/jpeg;base64,' + base64.b64encode(img_byte_arr).decode('utf-8')

        return jsonify({'predicted_class': predicted_class, 'predicted_color': predicted_color, 'image': img_str})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
