import cv2
import numpy as np
import random
import json
import os
import tempfile
from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
from craft_text_detector import detect_text_from_image
from PIL import Image

app = Flask(__name__)
CORS(app)
import cv2
import numpy as np

def deskew_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)  # Invert the image
    
    # Blur and apply threshold to get binary image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Get coordinates of non-zero pixels
    coords = np.column_stack(np.where(thresh > 0))
    
    # Compute the angle of the minimum area rectangle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Fix the angle to ensure correct orientation
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle  # Handle cases where the angle could cause 90-degree tilt
    else:
        angle = -angle
    
    # Get image center and create the rotation matrix
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the image
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def draw_bounding_boxes(image, lines):
    image = image.copy()
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in lines]
    for line_idx, line in enumerate(lines):
        color = colors[line_idx]
        for poly in line:
            pts = poly.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
    return image

import numpy as np


def process_polys_to_lines(polys, line_threshold_factor=0.60, column_threshold_factor=1.6):
    # Compute the average y-separation between words to infer if a word belongs to the same line
    heights = [np.max(p[:, 1]) - np.min(p[:, 1]) for p in polys]
    widths = [np.max(p[:, 0]) - np.min(p[:, 0]) for p in polys]
    avg_height = np.mean(heights)
    avg_width = np.mean(widths)
    line_threshold = avg_height * line_threshold_factor
    column_threshold = avg_width * column_threshold_factor
    
    # Sort polygons by their x-axis (left-to-right order)
    polys = sorted(polys, key=lambda p: np.min(p[:, 0]))
    output = []
    columns = []

    # Iterate over each polygon (word) in the document
    for poly in polys:
        y_min = np.min(poly[:, 1])
        x_min = np.min(poly[:, 0])
        x_max = np.max(poly[:, 0])
        added_to_line = False
        min_x_distance = float('inf')
        closest_column = None

        # Try to find the best column for this polygon
        for column in columns:
            column_min_x = np.min([np.min(line[0][:, 0]) for line in column])  # The min x of the leftmost line in the column

            # Check all lines in the column to see if this poly fits into one
            for line in column:
                last_poly_in_line = line[-1]  # Get the last polygon in the current line
                last_y_min_in_line = np.min(last_poly_in_line[:, 1])
                last_x_max_in_line = np.max(last_poly_in_line[:, 0])

                # Calculate the distances
                y_distance = abs(y_min - last_y_min_in_line)
                x_distance = x_min - last_x_max_in_line

                # Check if the current poly belongs to the same line (y-proximity)
                if y_distance <= line_threshold:
                    # Check if the poly belongs to the same column (x-proximity)
                    if x_distance <= column_threshold:
                        # Add to the same line in the column
                        line.append(poly)
                        added_to_line = True
                        break
            
            # If not added to any line, check if the poly is closer to this column based on column's min_x
            if not added_to_line:
                x_distance_to_column = x_min - column_min_x
                if x_distance_to_column < min_x_distance:
                    min_x_distance = x_distance_to_column
                    closest_column = column

            if added_to_line:
                break

        # If the polygon doesn't fit in any existing line, check if it belongs to a new line in the closest column
        if not added_to_line:
            if closest_column is not None and min_x_distance <= column_threshold:
                # Add as a new line to the closest column
                closest_column.append([poly])
            else:
                # Create a new column with the polygon starting a new line
                columns.append([[poly]])

    for column in columns:
        # Sort each column's lines by their vertical position (y-axis)
        column = sorted(column, key=lambda line: np.min(line[0][:, 1]))
        output.extend(column)
    
    return output

# Server
@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files.get('image')
    options_str = request.form.get('options')
    
    if not file:
        return jsonify({"error": "No image provided"}), 400
    
    if options_str:
        try:
            options = json.loads(options_str)
        except ValueError:
            return jsonify({"error": "Invalid JSON in options"}), 400
    else:
        options = {}

    return_image = options.get('return_image', False)
    return_coords = options.get('return_coords', False)

    image = np.array(Image.open(file.stream))

    # Preprocessing image from RGBA to RGB
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = deskew_image(image)
    _, polys, _ = detect_text_from_image(image, trained_model_path='./craft_mlt_25k.pth', use_cuda=False)
    lines = process_polys_to_lines(polys)

    response_data = {}
    if return_coords:
        line_coords = [{"line": idx + 1, 
                        "bounding_boxes": [
                            {"x_min": int(np.min(poly[:, 0])), 
                             "y_min": int(np.min(poly[:, 1])), 
                             "x_max": int(np.max(poly[:, 0])), 
                             "y_max": int(np.max(poly[:, 1]))} for poly in line]} 
                       for idx, line in enumerate(lines)]
        response_data["lines"] = line_coords

    if return_image:
        result_image = draw_bounding_boxes(image, lines)
        result_image_pil = Image.fromarray(result_image)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        result_image_pil.save(temp_file.name)

        image_url = url_for('serve_image', filename=os.path.basename(temp_file.name), _external=True)
        response_data["image_url"] = image_url

    if not response_data:
        return jsonify({"error": "No valid return option selected. Use return_image or return_coords."}), 400

    return jsonify(response_data)

@app.route('/image/<filename>', methods=['GET'])
def serve_image(filename):
    return send_file(os.path.join(tempfile.gettempdir(), filename), mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
