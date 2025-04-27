from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the YOLO model
model = YOLO("best.pt")
class_names = model.names

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        results = model.predict(img, conf=0.5)
        
        detections = []
        for r in results:
            boxes = r.boxes
            masks = r.masks
            
            if masks is not None:
                masks = masks.data.cpu()
                for i, (seg, box) in enumerate(zip(masks.data.cpu().numpy(), boxes)):
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    class_name = class_names[class_id]
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    seg = cv2.resize(seg, (img.shape[1], img.shape[0]))
                    contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        contour = max(contours, key=cv2.contourArea)
                        contour_points = contour.squeeze().tolist()
                    else:
                        contour_points = []
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'contour': contour_points
                    }
                    detections.append(detection)
        
        return jsonify({
            'detections': detections
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

# ⚡ Important: export the app for Vercel
app = app
