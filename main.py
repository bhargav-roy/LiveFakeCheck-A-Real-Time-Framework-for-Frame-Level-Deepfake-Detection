import sys, cv2, torch, random
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from model import Detector
from retinaface.pre_trained_models import get_model
from preprocess import extract_face
import warnings

warnings.filterwarnings("ignore")


class DeepfakeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiveFakeCheck")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.cam_selector = QComboBox()
        self.btn_start = QPushButton("Start")
        self.status_label = QLabel("Status: N/A")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Select Camera:"))
        controls.addWidget(self.cam_selector)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.status_label)
        layout.addLayout(controls)
        self.setLayout(layout)

        # Connect signals
        self.btn_start.clicked.connect(self.start_camera)

        # Timer and state
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.processing = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detect_cameras()

        # Load model
        self.model = Detector().to(self.device)
        checkpoint = torch.load("weights/FFraw.tar", map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        # Load face detector
        self.face_detector = get_model("resnet50_2020-07-20", max_size=1024, device=self.device)
        self.face_detector.eval()

    def detect_cameras(self):
        self.cam_selector.clear()
        for i in range(10):  # Check first 10 indexes
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                self.cam_selector.addItem(f"Camera {i}", i)
                cap.release()

    def start_camera(self):
        cam_index = self.cam_selector.currentData()
        self.cap = cv2.VideoCapture(cam_index)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Show video feed
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_frame, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        if not self.processing:
            self.processing = True
            self.run_inference(frame)

    def run_inference(self, frame):
        def process():
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = extract_face(img_rgb, self.face_detector)
                if not faces:
                    self.status_label.setText("Status: No Face")
                    self.processing = False
                    return

                img = torch.tensor(faces).to(self.device).float() / 255
                pred = self.model(img).softmax(1)[:, 1].cpu().data.numpy().tolist()
                fakeness = max(pred)

                result = "FAKE" if fakeness > 0.5 else "REAL"
                self.status_label.setText(f"Status: {result} ({fakeness:.2f})")

            except Exception as e:
                self.status_label.setText("Error during inference")
                print("Inference error:", e)
            finally:
                self.processing = False

        from threading import Thread
        Thread(target=process).start()


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    app = QApplication(sys.argv)
    window = DeepfakeApp()
    window.show()
    sys.exit(app.exec_())
