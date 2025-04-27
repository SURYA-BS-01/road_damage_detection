from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
from collections import defaultdict

# Load a model
model = YOLO("best.pt")
class_names = model.names
cap = cv2.VideoCapture('sample1.mp4')
count = 0

# Parameters for detection
conf_threshold = 0.5  # Confidence threshold
skip_frames = 2  # Process every nth frame
prev_detections = {}  # Store previous detections
tracking_history = defaultdict(list)  # Track object positions over time
max_track_history = 5  # Number of frames to consider for smoothing
next_id = 0  # For simple object tracking

# Function to calculate IOU between two boxes
def calculate_iou(box1, box2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

while True:
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    if count % skip_frames != 0:
        continue
    
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img, conf=conf_threshold)
    
    current_detections = {}
    
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
    
    if masks is not None:
        masks = masks.data.cpu()
        for i, (seg, box) in enumerate(zip(masks.data.cpu().numpy(), boxes)):
            confidence = float(box.conf)
            if confidence < conf_threshold:
                continue
                
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours, continue
            if not contours:
                continue
                
            contour = max(contours, key=cv2.contourArea)  # Use largest contour
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            box_coords = (x, y, x + w_rect, y + h_rect)
            
            # Simple object tracking
            matched_id = None
            for obj_id, prev_box in prev_detections.items():
                iou = calculate_iou(box_coords, prev_box[:4])
                if iou > 0.3:  # IOU threshold for same object
                    matched_id = obj_id
                    break
            
            if matched_id is None:
                matched_id = next_id
                next_id += 1
            
            d = int(box.cls)
            c = class_names[d]
            
            # Store current detection
            current_detections[matched_id] = (*box_coords, d, c, confidence)
            
            # Update tracking history
            tracking_history[matched_id].append(box_coords)
            if len(tracking_history[matched_id]) > max_track_history:
                tracking_history[matched_id].pop(0)
            
            # Smooth box position based on tracking history
            if len(tracking_history[matched_id]) > 1:
                avg_x = sum(box[0] for box in tracking_history[matched_id]) / len(tracking_history[matched_id])
                avg_y = sum(box[1] for box in tracking_history[matched_id]) / len(tracking_history[matched_id])
                avg_x2 = sum(box[2] for box in tracking_history[matched_id]) / len(tracking_history[matched_id])
                avg_y2 = sum(box[3] for box in tracking_history[matched_id]) / len(tracking_history[matched_id])
                
                # Use smoothed coordinates but keep some responsiveness with weighted average
                weight = 0.7  # Higher weight gives more importance to current position
                smooth_x = int(weight * box_coords[0] + (1-weight) * avg_x)
                smooth_y = int(weight * box_coords[1] + (1-weight) * avg_y)
                smooth_x2 = int(weight * box_coords[2] + (1-weight) * avg_x2)
                smooth_y2 = int(weight * box_coords[3] + (1-weight) * avg_y2)
                
                display_box = (smooth_x, smooth_y, smooth_x2, smooth_y2)
            else:
                display_box = box_coords
            
            # Draw contour
            cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
            
            # Draw smoothed bounding box
            x1, y1, x2, y2 = display_box
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Display class name with confidence
            label = f"{c}: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Update previous detections
    prev_detections = current_detections
                 
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
