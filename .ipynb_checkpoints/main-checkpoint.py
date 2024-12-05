from flask import Flask, request, jsonify
from ultralytics import YOLO
import threading
import time
import os
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)

# Global variables
model = None
model_lock = threading.Lock()
model_metrics = {}
model_build_date = None
data_yaml_path = os.path.expanduser('~/aiflow/dataset/data.yaml')
model_path = 'runs/detect/train/weights/best.pt'

def load_model(model_path):
    global model, model_build_date
    with model_lock:
        model = YOLO(model_path)
        model_build_date = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Model loaded: {model_path} at {model_build_date}")

def get_dataset_info():
    with open(data_yaml_path, 'r') as file:
        data_yaml = yaml.safe_load(file)
    return {
        'train': data_yaml.get('train'),
        'val': data_yaml.get('val'),
        'nc': data_yaml.get('nc'),
        'names': data_yaml.get('names'),
    }


def update_model_metrics():
    results_file = 'runs/detect/train/results.csv'
    if os.path.exists(results_file):
        import pandas as pd
        df = pd.read_csv(results_file)
        last_row = df.iloc[-1]
        global model_metrics
        model_metrics = {
            'epoch': int(last_row['epoch']),
            'time': float(last_row['time']),
            'metrics/precision(B)': float(last_row['metrics/precision(B)']),
            'metrics/recall(B)': float(last_row['metrics/recall(B)']),
            'metrics/mAP50(B)': float(last_row['metrics/mAP50(B)']),
            'metrics/mAP50-95(B)': float(last_row['metrics/mAP50-95(B)']),
            'train/box_loss': float(last_row['train/box_loss']),
            'train/cls_loss': float(last_row['train/cls_loss']),
            'train/dfl_loss': float(last_row['train/dfl_loss']),
            'val/box_loss': float(last_row['val/box_loss']),
            'val/cls_loss': float(last_row['val/cls_loss']),
            'val/dfl_loss': float(last_row['val/dfl_loss']),
            'lr/pg0': float(last_row['lr/pg0']),
            'lr/pg1': float(last_row['lr/pg1']),
            'lr/pg2': float(last_row['lr/pg2']),
        }
        print("Model metrics updated.")
    else:
        print("Results file not found. Cannot update model metrics.")

def retrain_model():
    # Run the training command
    training_command = (
        f"yolo task=detect mode=train model=yolo11m.pt "
        f"data={data_yaml_path} epochs=10 imgsz=640 plots=True"
    )
    os.system(training_command)

    # Load the new model
    if os.path.exists(model_path):
        load_model(model_path)
        update_model_metrics()
    else:
        print("Retraining failed or model not found.")

class DataYAMLChangeHandler(FileSystemEventHandler):
    def __init__(self, filepath):
        super(DataYAMLChangeHandler, self).__init__()
        self.filepath = filepath

    def on_modified(self, event):
        if event.src_path == self.filepath:
            print(f"Detected change in {self.filepath}. Retraining the model...")
            retrain_model()

def start_data_yaml_monitor():
    event_handler = DataYAMLChangeHandler(data_yaml_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(data_yaml_path), recursive=False)
    observer.start()
    print(f"Started monitoring {data_yaml_path} for changes.")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty image filename'}), 400

    temp_image_path = os.path.join('/tmp', image_file.filename)
    image_file.save(temp_image_path)

    # Perform inference
    with model_lock:
        results = model(temp_image_path)
    # Process the results
    detection_metadata = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detection = {
                'class_id': int(box.cls),
                'class_name': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
            }
            detection_metadata.append(detection)

    # Clean up
    os.remove(temp_image_path)

    return jsonify({'detections': detection_metadata})

@app.route('/reports', methods=['GET'])
def reports():
    report_data = {
        'model_build_date': model_build_date,
        'model_metrics': model_metrics,
        'dataset_info': get_dataset_info(),
    }
    return jsonify(report_data)

if __name__ == '__main__':
    load_model(model_path)  # Adjust with your initial model path
    start_data_yaml_monitor()
    app.run(host='0.0.0.0', port=5005)