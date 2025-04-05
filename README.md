# Acoustic UAV Detection System

## Overview

This project develops an innovative acoustic-based system for detecting unmanned aerial vehicles (UAVs) through analysis of their unique sound signatures. Using microphones and acoustic sensors, the system captures drone noises in various environments and employs advanced signal processing and machine learning techniques to accurately distinguish drone sounds from background noise. Additionally, the system estimates the direction and distance of detected drones using sound intensity analysis and time-delay localization methods, providing critical spatial information for timely responses.

## Features

- **Acoustic Sensing**: Utilizes microphones and acoustic sensors to capture UAV sound signatures.
- **Signal Processing**: Applies filters and spectral analysis to preprocess audio data and enhance drone sound characteristics.
- **Machine Learning**: Implements classification models (e.g., SVM, CNN) to differentiate drone noises from ambient sounds.
- **Localization**: Estimates direction and distance of drones using time-delay of arrival (TDOA) and sound intensity analysis.
- **Environment Adaptability**: Evaluates performance across diverse settings such as urban areas, event venues, and sensitive locations.

## Architecture

```
+----------------+       +----------------+       +-----------------+
| Acoustic Input | --->  | Preprocessing  | --->  | Feature Extract |
|  (Microphones) |       | (Filtering,    |       | (MFCC, Spectra) |
+----------------+       |  Noise Supp.)  |       +-----------------+
                            |
                            v
                     +-----------------+
                     | Classification  |
                     |  (ML Models)    |
                     +-----------------+
                            |
                            v
                     +-----------------+
                     | Localization    |
                     | (TDOA, Intensity)|
                     +-----------------+
                            |
                            v
                     +-----------------+
                     | Visualization & |
                     |   Reporting     |
                     +-----------------+
```

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - numpy
  - scipy
  - scikit-learn
  - librosa
  - sounddevice
  - matplotlib

### Installation

```bash
git clone https://github.com/yourusername/acoustic-uav-detection.git
cd acoustic-uav-detection
pip install -r requirements.txt
```

## Usage

1. **Data Collection**: Configure microphones and run `record_audio.py` to capture environmental and UAV audio samples.
2. **Preprocessing**: Execute `preprocess.py` to apply noise reduction and filtering.
3. **Training**: Use `train_model.py` to train classification models on labeled audio features.
4. **Detection**: Launch `detect_uav.py` for real-time UAV detection and localization.
5. **Visualization**: Run `visualize_results.py` to view detection logs and spatial plots.

## Project Structure

```
├── data/
│   ├── raw/           # Raw audio recordings
│   └── processed/     # Preprocessed audio files
├── src/
│   ├── record_audio.py
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── detect_uav.py
│   └── visualize_results.py
├── models/            # Saved ML models
├── requirements.txt
└── README.md
```

## Evaluation

The system is tested in multiple scenarios to assess detection accuracy, localization error, and robustness against background noise. Performance metrics such as precision, recall, F1-score, and mean localization error are reported.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, feature enhancements, or documentation improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

