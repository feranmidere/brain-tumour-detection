# Brain Tumour Detection Pipeline & Reactive Web Application

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1%2B-EE4C2C?logo=pytorch)
![Solara](https://img.shields.io/badge/Solara-1.18.0%2B-FF4B4B?logo=python)
![ipywidgets](https://img.shields.io/badge/ipywidgets-8.1.0%2B-FFA000?logo=jupyter)
![Docker](https://img.shields.io/badge/Docker-24.0.0%2B-2496ED?logo=docker)

An end-to-end Deep Learning and computer vision application designed to detect brain tumours from MRI scans using the ResNet50 architecture. This repository features a production-ready ML workflow—from a scratch-trained PyTorch model designed specifically for medical imaging to a reactive, containerized web user interface built with **Solara** and **ipywidgets**.

---

## Architecture & Key Features

- **Custom PyTorch Training Loop**: Engineered a custom PyTorch training pipeline from scratch for a ResNet50 model.
- **Domain Shift Mitigation**: Intentionally omitted ImageNet pretrained weights to train directly on medical MRI scans, mitigating domain shift from natural image distributions.
- **Class Imbalance & Validation Strategy**: Validated across stratified splits using precision-recall analysis to account for class imbalance, achieving an impressive **0.98 PR-AUC**.
- **Reactive Web UI**: Built an interactive web frontend powered by **Solara** and **ipywidgets** allowing real-time image uploads, interactive visual feedback, and instant model predictions.
- **Containerized Deployment**: Fully packaged with **Docker** for reproducible, seamless execution across local and cloud environments.
- **Dataset**: Trained and evaluated on the benchmark [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset).

---

## Tech Stack & Dependencies

- **Language**: `python >= 3.10`
- **Deep Learning & Computer Vision**:
  - `torch >= 2.0.1`
  - `torchvision >= 0.15.2`
  - `opencv-python >= 4.8.0`
  - `scikit-learn >= 1.3.0`
  - `numpy >= 1.24.3`
  - `pillow >= 10.0.0`
- **Reactive Web Application**:
  - `solara >= 1.18.0`
  - `ipywidgets >= 8.1.0`
  - `matplotlib >= 3.7.2`
- **Containerization**:
  - `docker >= 24.0.0`

---

## Quickstart & Running the App

### Option 1: Running with Docker (Recommended)

1. **Build the Docker image:**
```bash
docker build -t brain-tumour-detection
```

2. **Run the container:**
```bash
docker run -p 8765:8765 brain-tumour-detection
```

3. **Open the Web UI:**
   Navigate to `http://localhost:8765` in your browser.

---

### Option 2: Local Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/feranmidere/brain-tumour-detection.git](https://github.com/feranmidere/brain-tumour-detection.git)
cd brain-tumour-detection
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -e .
```

4. **Launch the Solara application:**
```bash
solara run main.py:page --port 8765
```

Alternatively, use the `inference.ipynb` notebook in the notebooks folder to run the inference UI in jupyter.

---

## Dataset & Training Methodology

The model was trained and evaluated on the [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset).

### Why Train ResNet50 From Scratch?
Standard transfer learning relies on ImageNet pretrained weights, which are derived from natural everyday objects (animals, vehicles, items). Applying these features directly to specialized medical imaging can lead to suboptimal feature extraction due to severe **domain shift**.

To overcome this:
- **Scratch Architecture**: A ResNet50 model was trained completely from scratch, forcing the kernels to learn low-level and high-level medical imaging features (such as subtle tissue density variations and structural anomalies) directly from brain MRI scans.
- **Evaluation Metric**: Due to target class imbalance in medical diagnostics, standard accuracy can be misleading. Model evaluation relied on **Precision-Recall Area Under Curve (PR-AUC)** across stratified splits, achieving a final validation **PR-AUC of 0.98**.

---

## Interactive Web Interface

The interactive frontend is built natively in Python using **Solara** and **ipywidgets**:

- **File Upload Widget**: Drag and drop brain MRI image files directly into the web app.
- **Reactive Rendering**: State changes dynamically update the visual preview of the uploaded MRI and trigger real-time inference without full page reloads.
- **Prediction Analysis**: Displays classification outcomes along with prediction confidence scores and target class probability distributions.

---

## Author

**Feranmi**
- GitHub: [@feranmidere](https://github.com/feranmidere)
