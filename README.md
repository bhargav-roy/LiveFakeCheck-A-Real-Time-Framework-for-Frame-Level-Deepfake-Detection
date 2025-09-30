# LiveFakeCheck – A Real-Time Framework for Frame-Level Deepfake Detection

A **PyQt5-based application** for testing deepfake detection on live video streams using pretrained model weights.

![LiveFakeCheck](https://github.com/user-attachments/assets/875ed71f-0eb4-4c63-aca8-00097e7d11ba)

This application is built on top of the [**Self-Blended Images (SBI)**](https://github.com/mapooon/SelfBlendedImages) method to classify live video frames as real or fake.

---

## 🔧 Installation

To create a conda environment, run:

```bash
conda create -n lfcenv python=3.9 -y
```

To activate conda environment, run:

```bash
conda activate lfcenv
```

To install the required dependencies, run:

```bash
pip3 install -r requirements.txt
```

---

## 🚀 Running the Application

1. Download the pretrained model weights provided by the SBI authors [**here**](https://github.com/mapooon/SelfBlendedImages?tab=readme-ov-file#2-pretrained-model).
2. Place the downloaded weights in the `./weights/` directory.
3. Launch the application:

```bash
python main.py
```

---

## 🧪 Compatible Deepfake Software (for testing)

- **DeepFaceLive**: [GitHub Link](https://github.com/iperov/DeepFaceLive)
- **Avatarify Desktop (CUDA ≥ 11.8)**: [GitHub Link](https://github.com/tom-n96/avatarify-desktop-cuda-11.8)

---

## 📷 Virtual Camera Setup (Optional)

To route deepfake software output into this app:

- **OBS Studio** (for virtual webcam): [obsproject.com](https://obsproject.com/)

---

## 📌 Notes

- This framework currently supports frame-level models. Future versions may support optical flow–based or hybrid models.
- Preprocessing sensitivity (e.g., sharpening) significantly impacts detection performance.
