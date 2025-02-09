from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  
import tensorflow as tf
import numpy as np
import cv2
import os
import json
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import ollama
from collections import Counter

# Initialize Flask app
# app = Flask(__name__, template_folder=os.path.join("backend", "templates"), static_folder=os.path.join("backend", "static"))
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable CORS for all routes
client = ollama.Client()

# Define paths
CLASSIFICATION_MODEL_DIR = os.path.join("models", "ensemble_model")
CLASS_LABELS_PATH = os.path.join("data", "class_labels.json")
SEGMENTATION_MODEL_PATH = os.path.join("models", "yolo_best.pt")
UPLOAD_FOLDER = os.path.join("data", "uploads")
LEAVES_FOLDER = os.path.join("data", "leaves")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class labels
with open(CLASS_LABELS_PATH, "r") as json_file:
    class_labels = json.load(json_file)

# Load models
segmentation_model = YOLO(SEGMENTATION_MODEL_PATH)
classification_model = tf.saved_model.load(CLASSIFICATION_MODEL_DIR)
inference_func = classification_model.signatures["serving_default"]

def convert_to_jpg(img_path):
    image = cv2.imread(img_path)  # Read the image
    new_path = os.path.splitext(img_path)[0] + ".jpg"  # Change extension to .jpg
    cv2.imwrite(new_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # Save as JPG
    return new_path

def resize_with_padding(image, target_size=(256, 256), padding_color=(0, 0, 0)):
    """
    Resize an image while maintaining aspect ratio by adding padding.
    
    Args:
        image: Input image (numpy array).
        target_size: Desired output size (width, height).
        padding_color: Color of the padding (BGR format).
    
    Returns:
        Padded and resized image.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale to fit within target size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image while maintaining aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding (equal on both sides if necessary)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # Apply padding to center the image
    padded = cv2.copyMakeBorder(
        resized, pad_h, target_h - new_h - pad_h, pad_w, target_w - new_w - pad_w,
        borderType=cv2.BORDER_CONSTANT, value=padding_color
    )

    return padded

def extract_leaves(results, image_path, output_dir=LEAVES_FOLDER):
    """
    Extracts segmented leaves from a YOLO model's results, saves them with a gray background,
    and returns the full file paths of the cropped leaves. If no leaves are detected, returns
    the original image path in a list.

    Args:
        results (list): YOLO model predictions.
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the extracted leaves.

    Returns:
        list: List of file paths for saved leaf images. If no detections, returns [image_path].
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load original image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (480, 640))

    # Handle case where no masks are detected
    if not hasattr(results[0], "masks") or results[0].masks is None:
        print("No mask detected. Returning original image.")
        return [image_path]
    
    saved_paths = []  # List to store output file paths
    
    # Get the masks
    mask_array = results[0].masks.data.cpu().numpy()  # (num_objects, height, width)
    
    # Loop through results (assuming single image inference)
    for i, mask in enumerate(mask_array):
        
        # Extract the leaf    
        mask = mask.astype(np.uint8) # Clip to 0-255
        if mask.shape[:2] != image.shape[:2]: # Ensure same size
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = cv2.bitwise_and(image, image, mask=mask) # Overlap with mask
        
        # Crop the result to the get exaclty the leaf while maintaining the leaf's proportion
        coords = cv2.findNonZero(mask) # Get coordinates of non-black pixels
        x, y, w, h = cv2.boundingRect(coords) # Get bounding box
        cropped_result = result[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
        cropped_result = resize_with_padding(cropped_result)
        
        # Generate output file path
        output_filename = f"{os.path.basename(image_path).replace('.jpg', '')}_leaf_{i}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the cropped leaf with gray background
        cv2.imwrite(output_path, cropped_result)

        # Store saved file path
        saved_paths.append(output_path)

    return saved_paths if len(saved_paths)!=0 else [image_path]  # Return original image if no leaves were saved

def segment_image(img_path):
    """ Run segmentation model to extract the region of interest (ROI) """
    if os.path.splitext(img_path)[1].lower() != ".jpg":
        img_path = convert_to_jpg(img_path) 
    results = segmentation_model.predict(source=img_path, conf=0.5, save=True, exist_ok=True)
    segmented_image_path = extract_leaves(results, img_path)
    return segmented_image_path

def load_image(img_path):
    """ Preprocess segmented image for classification """
    img_size = 256
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)  
    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img  # Adjust for ensemble model

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/analyze")
def analyze():
    return render_template("analyze.html")

@app.route("/about")
def about():
    section = request.args.get('section', 'default')  
    return render_template("about.html", section=section)

@app.route("/classify", methods=["POST"])
def classify():
    """ API endpoint to classify an uploaded image after segmentation """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # Run segmentation model
    segmented_image_paths = segment_image(file_path)
    results = []
    predictions = []
    for segmented_image_path in segmented_image_paths:
        # Preprocess and classify segmented image
        input_data = load_image(segmented_image_path)
        prediction = inference_func(tf.constant(input_data))
        predictions_array = prediction[list(prediction.keys())[0]].numpy()
        predicted_class_index = np.argmax(predictions_array)
        predicted_label = class_labels[predicted_class_index]
        predictions.append(predicted_label)
        # Store the result
        results.append({
            "file_path": segmented_image_path,
            "prediction": predicted_label
        })

    # Save to results.json
    with open("results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    return jsonify({"prediction": Counter(predictions).most_common(1)[0][0]})

@app.route("/generate", methods=["POST"])
def generate():
    """ API endpoint to generate additional insights using Ollama """
    data = request.get_json()
    if "prediction" not in data:
        return jsonify({"error": "No prediction provided"}), 400
    
    predicted_label = data["prediction"]
    predicted_label = str(predicted_label).replace('[','').replace(']','')
    response = client.generate("plantidoc", predicted_label)
    return jsonify({"prediction": predicted_label, "ollama_response": response.response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
