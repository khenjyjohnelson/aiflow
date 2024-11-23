from ultralytics import YOLO
from roboflow import Roboflow

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights


rf = Roboflow(api_key="J0XF1zhI2TuRJarkeWSv")
project = rf.workspace("tomato-leaf-disease-6wx9g").project("tomato-leaf-disease-by-vaskor")
version = project.version(2)
dataset = version.download("yolov11")
                

# Train the model
results = model.train(data=dataset, epochs=100, imgsz=640)