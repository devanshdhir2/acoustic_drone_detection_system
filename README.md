# Acoustic-Based Drone Detection System 🚁📡

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Hardware](https://img.shields.io/badge/Hardware-Raspberry_Pi_5-red)

**Real-Time UAV Identification and Localization using Edge AI**

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Performance](#-results--performance)
- [Team & Acknowledgements](#-team--acknowledgements)
- [License](#-license)

---

## 📝 About the Project

Unmanned Aerial Vehicles (UAVs) pose increasing security risks to critical infrastructure and personal privacy. Traditional detection techniques such as radar and vision-based systems often struggle with small consumer drones, cluttered environments, or low-light conditions.

**Acoustic Drone Detection System** designed for real-time airspace monitoring. It uses a **Raspberry Pi 5** and a **Seeed Studio ReSpeaker 4-Mic Array** to capture the unique acoustic signatures of drone motors. A lightweight **Random Forest** machine-learning model classifies incoming audio and estimates the **Direction of Arrival (DOA)**, enabling rapid localization. The entire system runs **offline at the edge**, ensuring privacy and low latency.

---

## ✨ Key Features

- **🌲 Edge AI Core:** Optimized Random Forest classifier achieving **~96% accuracy** with sub-350 ms inference latency.
- **🧭 Real-Time Localization:** Estimates sound source direction (0°–360°) using Time Difference of Arrival (TDOA) from the 4-microphone array.
- **🔇 Intelligent Noise Filtering:** Differentiates drones from wind, traffic, and human speech using MFCC, spectral, and temporal features.
- **🖥️ Tactical Dashboard:** Standalone Python GUI with radar-style visualization and clear detection status (Red/Green).
- **🔒 Privacy-First:** Fully offline operation with no cloud dependency.
- **⚡ Automatic Gain Control (AGC):** Dynamically adjusts microphone sensitivity to detect distant drones.

---

## 🏗 System Architecture

The system follows a modular, real-time processing pipeline:

1. **Audio Capture:** Raw multi-channel audio input from the ReSpeaker 4-Mic Array
2. **Preprocessing:** Noise reduction, framing, normalization, and digital AGC
3. **Feature Extraction:** MFCCs, Spectral Contrast, Chroma, and Zero-Crossing Rate using `librosa`
4. **Inference Engine:** Random Forest model classifies audio as **Drone** or **Noise**
5. **Localization:** DOA algorithm estimates the angle of arrival when a drone is detected
6. **Visualization:** Radar dashboard updates with target position and confidence

---

## 🛠 Hardware Requirements

- **Single Board Computer:** Raspberry Pi 5 (8 GB RAM recommended)
- **Microphone Array:** Seeed Studio ReSpeaker 4-Mic Array (USB)
- **Power Supply:** USB-C PD 27 W
- **Cooling:** Raspberry Pi Active Cooler
- **Display:** HDMI monitor
- **Storage:** High-speed microSD card (32 GB or higher)

---

## 💻 Tech Stack

- **Programming Language:** Python 3.11
- **Machine Learning:** Scikit-learn, Joblib
- **Audio Processing:** Librosa, NumPy, PyAudio, SciPy
- **Hardware Interface:** Seeed Voicecard Drivers
- **Visualization & GUI:** Matplotlib (Animation API), Tkinter

---

## 📊 Dataset

A custom dataset was created using real-world recordings of commercial quadcopters and diverse environmental noise samples.

🔗 **Dataset:**  
https://www.kaggle.com/datasets/gautamdhawan55/merged-drone

**Dataset Structure**

- `drone/` – 148+ UAV motor sound samples
- `noise/` – 125+ ambient noise samples

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/acoustic-drone-detection.git
cd acoustic-drone-detection
```

### 2. Install System Dependencies (Raspberry Pi)

```bash
sudo apt update
sudo apt install python3-pyaudio portaudio19-dev libatlas-base-dev
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Phase 1: Model Training (Optional)

1. Place the dataset folder inside the project directory
2. Open `training.ipynb`
3. Run all cells to preprocess audio and train the model
4. Generated files:
   - `drone_brain_v2.pkl`
   - `feature_scaler.pkl`

---

### Phase 2: Real-Time Deployment

```bash
python3 main.py
```

The radar dashboard will launch and highlight detected drones in **red** with direction locking.

---

## 📈 Results & Performance

- **Classification Accuracy:** 96.32%
- **Inference Latency:** < 350 ms per audio chunk
- **Detection Range:** ~10 meters
- **Localization Accuracy:** ±15° error
- **Thermal Stability:** CPU temperature < 65 °C

---

## 👥 Team & Acknowledgements

**Capstone Project (CPG-179)**  
**Thapar Institute of Engineering & Technology, Patiala**

**Team Members**

- Miet Pamecha (102203012)
- Gautam Dhawan (102203061)
- Lipsita Devgan (102203408)
- Tamanna Bajaj (102203413)
- Devansh Dhir (102203449)

**Faculty Mentor**

- **Dr. Sharad Saxena**  
  Associate Professor, Department of Computer Science & Engineering

---
