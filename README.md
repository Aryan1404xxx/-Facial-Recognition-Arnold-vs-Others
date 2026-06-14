# 🎭 Facial Recognition: Arnold vs Others

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)
![Gradio](https://img.shields.io/badge/Gradio-6.18-green)
![Accuracy](https://img.shields.io/badge/Accuracy-94%25-brightgreen)

A supervised machine learning project that identifies whether a face image is **Arnold Schwarzenegger** or someone else, built with Python, scikit-learn, and a Gradio web interface.

## 📸 Demo
Upload any face image and the model will predict whether it's Arnold or not, along with a confidence percentage!

## 🧠 How It Works
1. Dataset is loaded from **Labeled Faces in the Wild (LFW)** via scikit-learn
2. Data is balanced (42 Arnold vs 42 Others) to avoid bias
3. **PCA (Eigenfaces)** reduces image dimensions to 30 components
4. **SVM with RBF kernel** classifies the face
5. Result is shown via a **Gradio web interface**

## 📊 Model Performance
| Metric | Arnold | Others |
|--------|--------|--------|
| Precision | 100% | 89% |
| Recall | 89% | 100% |
| F1-Score | 94% | 94% |
| **Accuracy** | **94%** | |

## 🚀 How to Run

### 1. Clone the repo
\`\`\`bash
git clone https://github.com/Aryan1404xxx/-Facial-Recognition-Arnold-vs-Others.git
cd -Facial-Recognition-Arnold-vs-Others
\`\`\`

### 2. Create virtual environment
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
\`\`\`

### 3. Install dependencies
\`\`\`bash
pip install scikit-learn opencv-python numpy matplotlib gradio
\`\`\`

### 4. Train the model
\`\`\`bash
python train.py
\`\`\`

### 5. Launch the app
\`\`\`bash
python app.py
\`\`\`

### 6. Open in browser
\`\`\`
http://127.0.0.1:7860
\`\`\`

## 📁 Project Structure
\`\`\`
facial_recognition/
├── app.py          # Gradio web interface
├── train.py        # Model training script
├── .gitignore      # Git ignore file
└── model/
    └── model.pkl   # Saved trained model (auto-generated)
\`\`\`

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| scikit-learn | ML model + LFW dataset |
| OpenCV | Image processing |
| Gradio | Web interface |
| NumPy | Data manipulation |
| PCA | Dimensionality reduction |
| SVM | Classification |

## 📌 Notes
- Built and tested on **MacBook Air M2**
- Dataset auto-downloads via scikit-learn (no manual setup)
- Model is saved as `model.pkl` after training
- Balanced dataset used to avoid class imbalance

## 👤 Author
**Aryan Sinha** — [GitHub](https://github.com/Aryan1404xxx)
