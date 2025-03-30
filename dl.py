from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

!pip install roboflow

import ultralytics
from roboflow import Roboflow

# Authenticate and get the dataset
rf = Roboflow(api_key="8eMs5jw1EQwDHiEuNZwW")
project = rf.workspace("neural-ocean").project("neural_ocean")
version = project.version(3)
dataset = version.download("yolov11")

from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # You can load the base model or any custom model here

model.train(
    data='/content/drive/MyDrive/test/data.yaml',  # Correct path to your data.yaml in the uploaded folder
    epochs=10,                             # Training for 20 epochs
    imgsz=640,                              # Image size
    batch=16,                               # Customize batch size if necessary
    device='cpu',                           # Use CPU (can change this if GPU is available in Colab)
    save=True                               # Save the best model during training
)

source = '/content/drive/MyDrive/test/images/1bc7-iudfmpmn7245599_jpg.rf.0c27f6617b1c2d7665a4badbb8474e28.jpg'  # Replace with your test image
results = model.predict(
    source=source,
    conf=0.15,         # Confidence threshold for detection
    save=True          # Save results
)

# Load the trained model from train4 folder
model = YOLO('runs/detect/train2/weights/best.pt')

# Validate the model
results = model.val(
    data="/content/drive/MyDrive/test/data.yaml",  # Path to the dataset YAML file
    imgsz=640,                        # Image size
    batch=16,                         # Batch size
    device='cpu'                      # Use CPU (if no GPU available)
)

# Show validation results
from IPython.display import Image, display

files = [
    "runs/detect/train2/P_curve.png",
    "runs/detect/train2/R_curve.png",
    "runs/detect/train2/confusion_matrix.png",
    "runs/detect/train2/labels.jpg",
    "runs/detect/train2/results.png"
]

# Display each validation file if it exists
for file in files:
    try:
        display(Image(filename=file, width=600))
    except FileNotFoundError:
        print(f"File not found: {file}")

# Predict on test dataset
source = '/content/drive/MyDrive/test/images'  # Update with the correct folder for test images
results = model.predict(
    source=source,
    conf=0.15,       # Confidence threshold for detections
    save=True        # Save predictions
)

# Display the predicted test images
import glob
import os
from IPython.display import Image as IPyImage, display

latest_folder = max(glob.glob('runs/detect/predict*/'), key=os.path.getmtime)
for img in glob.glob(f'{latest_folder}/*.jpg')[:10]:  # Display top 10 predictions
    display(IPyImage(filename=img, width=600))
    print('\n')

# Optional: Test prediction on a single image
source = '/content/drive/MyDrive/test/images/1bc7-iudfmpmn7245599_jpg.rf.0c27f6617b1c2d7665a4badbb8474e28.jpg'  # Replace with the image you want to test
results = model.predict(
    source=source,
    conf=0.25,
    save=True
)

# Display the result for the single image
try:
    predicted_image = f"{latest_folder}/{os.path.basename(source)}"
    display(IPyImage(filename=predicted_image, width=600))
except FileNotFoundError:
    print(f"Prediction result not found for {source}")




