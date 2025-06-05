# Satellite Object Detection using YOLOv8

This project uses a custom-trained YOLOv8 model to detect multiple classes of objects from satellite images. The training was performed on a public dataset using the Roboflow platform, and the model is deployed via a Streamlit web app for easy image inference.

## Project Structure

```
cv_satellite_project/
  app.py                  
  best.pt                 
  requirements.txt        
  README.md               
  notebooks/
    yolov8_training_pipeline.ipynb  # Training pipeline notebook
```

## Features

- Custom object detection on satellite images
- 23 object classes (buildings, roads, rivers, playgrounds, etc.)
- Inference powered by YOLOv8 (Ultralytics)
- Streamlit frontend for easy image upload & prediction
- Fully containerized and GitHub-ready project structure

## Installation

### Clone the repository:

```
git clone https://github.com/your-username/cv_satellite_project.git
cd cv_satellite_project
```

### Install dependencies:

```
pip install -r requirements.txt
```

### Run the Streamlit app:

```
streamlit run app.py
```

## Model Details

- Model: YOLOv8n (Nano version)
- Framework: Ultralytics YOLOv8
- Dataset: Satellite images annotated via Roboflow
- Classes: 23 satellite object classes
- Training: 50 epochs on Google Colab with GPU
