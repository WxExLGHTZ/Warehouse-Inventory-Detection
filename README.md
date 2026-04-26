# Warehouse Inventory Detection

A neural-network-based system that recognizes the contents of a 3×3 storage rack from camera images, maps each detected object to its grid position using bounding box coordinate analysis, and logs the inventory state to an Azure SQL database.

[!NOTE]
The model is trained on a specific type of storage rack and specific objects. Detection accuracy on other rack configurations or object types is not guaranteed.

## Motivation

Manually tracking warehouse inventory is error-prone and time-consuming. This system automates the process by pointing a camera (webcam or Intel RealSense) at a physical storage rack and using a fine-tuned SSD MobileNet V1 model to detect colored objects in each compartment. The key challenge is not just detecting objects but determining *which compartment* each object occupies — solved here by sorting detected bounding boxes by their spatial coordinates and mapping them onto a 3×3 grid. Results are visualized as a color-coded grid diagram and persisted to a database with timestamped error logs.

## Features

- Detects four object classes (red, blue, white, empty) in a 3×3 storage rack using a fine-tuned SSD MobileNet V1 FPN model
- Maps detected bounding boxes to grid positions (A1–C3) through spatial coordinate analysis
- Provides a PyQt5 desktop GUI with live camera feed, manual and automatic photo capture modes
- Generates a color-coded visualization of the recognized inventory state
- Logs inventory snapshots and error states to an Azure SQL Server database
- Supports both webcam and Intel RealSense D400 series cameras
- Includes the full training pipeline: data labeling with LabelImg, TFRecord generation, model fine-tuning, and TensorBoard evaluation
- Runs detection in a separate thread to keep the GUI responsive

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.8.x |
| Object Detection | TensorFlow 2.x Object Detection API |
| Pre-trained Model | SSD MobileNet V1 FPN 640×640 (COCO17) |
| Computer Vision | OpenCV |
| GUI | PyQt5 |
| Database | Azure SQL Server (via pyodbc, ODBC Driver 17) |
| Camera | Intel RealSense (pyrealsense2) / Webcam |
| Labeling | LabelImg |
| Serialization | Protocol Buffers (protoc 3.15.6) |

## Architecture

The system follows a multi-stage pipeline: the **Setup** module bootstraps the entire TensorFlow Object Detection API environment, including cloning the TF Model Garden, compiling protobuf definitions, and installing all dependencies. The **Training** module generates TFRecords from labeled images, fine-tunes the SSD MobileNet V1 FPN model on four custom classes, and evaluates the results via TensorBoard.

At runtime, the **Frontend** (PyQt5) captures frames from a webcam or RealSense camera and passes them to the **Recognition** module, which runs inference and performs the spatial grid mapping — it identifies the four corner compartments first by comparing bounding box coordinates, then locates the edge and center compartments relative to those anchors. The resulting 3×3 inventory array is handed to the **Visualization** module, which renders a matplotlib diagram and stores the result alongside an error log entry in the Azure SQL database.

```
Camera Frame → TF Detection → Bounding Box Sorting → 3×3 Grid Mapping → Visualization + DB Storage
```

## Getting Started

### Prerequisites

- Windows (the setup scripts use `cmd /c` commands)
- Python 3.8.x (tested with 3.8.0 and 3.8.10)
- PyCharm Community Edition (recommended)
- ODBC Driver 17 for SQL Server

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/warehouse-inventory-detection.git
   cd warehouse-inventory-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install the base dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install opencv-python
   pip install wget
   pip install gitpython
   pip install pyrealsense2
   ```

4. Open the project in PyCharm and set the interpreter to `venv/Scripts/Python.exe`.

5. Set `main.py` as the run configuration entry point and run it. This executes the **Setup**, which automatically downloads and installs the TensorFlow Object Detection API, protoc, and all remaining dependencies.

> [!IMPORTANT]
> The setup phase takes considerable time as it clones repositories, downloads model weights, and installs packages. It only needs to run once.

### Post-Setup Configuration

After the setup completes successfully, switch `main.py` from setup mode to application mode:

1. Comment out the setup imports and calls:
   ```python
   # from Backend_Pakete.setup import *
   # setup_obj = Setup()
   # setup_obj.setup_start()
   ```

2. Uncomment the application imports and calls:
   ```python
   from Frontend_Pakete.real_sense_main import *

   app = QApplication(sys.argv)
   main_window = MainWindow()
   main_window.show()
   sys.exit(app.exec_())
   ```

## Usage

1. Click **"Kamera verbinden"** to start the camera feed (defaults to webcam).
2. Click **"Foto machen"** to capture an image and run detection.
3. The recognized inventory appears as a color-coded grid on the right side of the GUI.
4. Results and logs are automatically stored in the database.

A default test image is provided at `Testimage/test.png` for running detection without a camera.

### Automatic Mode

Click **"Automatik starten"** to capture and analyze images every 2 minutes automatically.

### Using the RealSense Camera

To switch from webcam to a connected Intel RealSense camera, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the `control_timer` and `create_photo` methods of `real_sense_main.py`, and swap the thread target from `image2` (test image) to `image` (live frame).

### TensorBoard

To inspect training and evaluation metrics:

```bash
# Training metrics
cd Tensorflow/workspace/models/my_ssd_mobilenet/train
tensorboard --logdir=.

# Evaluation metrics
cd Tensorflow/workspace/models/my_ssd_mobilenet/eval
tensorboard --logdir=.
```


## Authors

- William Eppel
- Mert Karadeniz
- Viktor Kruckow

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
