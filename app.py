import cv2
import numpy as np
import random
import json
import os
import tempfile
from flask import Flask, request, jsonify, send_file, url_for
from craft_text_detector import detect_text_from_image
from PIL import Image

app = Flask(__name__)

def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
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

def process_polys_to_lines(polys, line_threshold_factor=0.60, column_threshold_factor=1.5):
    # Compute the avg y separation from each word to resonably infer if a word belongs to another line
    heights = [np.max(p[:, 1]) - np.min(p[:, 1]) for p in polys]
    avg_height = np.mean(heights)
    line_threshold = avg_height * line_threshold_factor
    column_threshold = column_threshold_factor * 10

    # Sort by x axis: Help us compare elements that are closer toguether along the x axis
    polys = sorted(polys, key=lambda p: np.min(p[:, 0]))
    lines = []

    # Iterate over each detected word in the document
    for poly in polys:
        y_min = np.min(poly[:, 1])
        x_min = np.min(poly[:, 0])
        added_to_line = False
        
        # Find the closest match based on the last word added to the line accoridng to it's x and y distance to the word
        for line in lines:
            last_poly_in_line = line[-1]
            last_y_min_in_line = np.min(last_poly_in_line[:, 1])
            last_x_max_in_line = np.max(last_poly_in_line[:, 0])
            y_distance_between_words = abs(y_min - last_y_min_in_line)
            x_distance_between_words = abs(x_min - last_x_max_in_line)
            if y_distance_between_words <= line_threshold and x_distance_between_words <= column_threshold:
                line.append(poly)
                added_to_line = True
                break
        
        # When no correlting line is found -> Initiate new line
        if not added_to_line:
            lines.append([poly])
    return lines

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
