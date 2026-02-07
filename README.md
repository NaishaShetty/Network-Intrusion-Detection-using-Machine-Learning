# 🛡️ Advanced Network Intrusion Detection System (NIDS)

An **end-to-end Machine Learning–powered Network Intrusion Detection System** built on the **NSL-KDD dataset**, designed to detect malicious network traffic with **high accuracy, calibrated probabilities, and real-time monitoring**.

This project combines a **high-performance FastAPI backend**, multiple state-of-the-art ML models, **probability calibration**, **concept drift detection**, and a **modern React-based security dashboard** for deep analysis and visualization.

---

## 🔗 Demo & Access URLs

> ⚠️ Demo Url
https://network-intrusion-detection-system-gp15.onrender.com

> ⚠️ Local Deployment (Default)

- **Frontend Dashboard:**  
  http://localhost:3000

- **Backend API:**  
  http://localhost:8000

- **Interactive API Docs (Swagger UI):**  
  http://localhost:8000/docs

---

## 🚀 Key Features

### 🔐 Backend (Intelligence Engine)

- **Multi-Model Architecture**
  - Decision Tree  
  - SGD Classifier  
  - Random Forest  
  - XGBoost  
  - LightGBM  

- **Automated Model Optimization**
  - Integrated `GridSearchCV` and `RandomizedSearchCV`
  - Model-specific hyperparameter tuning pipelines

- **Probability Calibration**
  - Uses `CalibratedClassifierCV` with **Isotonic Regression**
  - Ensures **statistically reliable probability scores**
  - Enables precise **threshold-based decision making**

- **Concept Drift Detection**
  - Compares live traffic distributions with training baselines
  - Detects evolving attack patterns over time

- **Security-Centric Metrics**
  - False Negative Rate (FNR)
  - Balanced Accuracy
  - ROC-AUC
  - Precision / Recall / F1 Score

- **Robust Data Preprocessing**
  - Automatic categorical encoding
  - Feature scaling
  - Data sanitization and validation

- **High-Performance API**
  - Built with **FastAPI**
  - Asynchronous inference endpoints
  - Low-latency prediction support

---

### 🧭 Frontend (Command Center)

- **Real-Time Training Monitor**
  - Live model training progress
  - Hyperparameter tuning & calibration status logs

- **Performance Deep Dive**
  - Confusion Matrices
  - ROC Curves
  - Accuracy, Precision, Recall & F1 comparisons
  - Radar charts for multi-metric evaluation

- **Security Insights Engine**
  - Automatically highlights the **most production-ready model**
  - Recommendations based on security posture (low FNR, high recall)

- **Interactive Prediction Lab**
  - Drag-and-drop CSV upload
  - Adjustable classification threshold slider
  - Attack vs Normal probability histograms

- **Operational Monitoring**
  - Per-model inference latency (ms)
  - Feature-level distribution shift tracking
  - System health indicators

- **Traffic Simulation**
  - Visual simulation of incoming network packets
  - Live intrusion detection alerts

---

## 🛠️ Tech Stack

### 🧠 Backend
- **Language:** Python 3.10+
- **Framework:** FastAPI
- **Machine Learning:**  
  - Scikit-learn  
  - XGBoost  
  - LightGBM
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Calibration:** CalibratedClassifierCV (Isotonic Regression)

---

### 🖥️ Frontend
- **Library:** React.js
- **Visualization:** Recharts
- **Styling:** Modern Vanilla CSS (Dark Mode + Glassmorphism)
- **Icons:** Lucide-React
- **API Client:** Axios

---

## 📄 How to Run the Project

### 🔧 Backend Setup
```bash
python -m uvicorn src.api:app --reload
```
Backend will now be available at: http://localhost:8000

### 🔧 Frontend Setup

- cd frontend
- npm install
- npm start

Frontend will now be available at: http://localhost:3000

# Training Models

- Open the Frontend Dashboard
- Navigate to Overview
- Click Train Models
- Monitor live training, tuning, and calibration progress

# Dataset

- NSL-KDD Dataset
- Widely used benchmark dataset for intrusion detection research
- Designed to address redundancy and imbalance issues in KDD’99
