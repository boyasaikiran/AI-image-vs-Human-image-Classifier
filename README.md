

# 🧠 AI vs Human Image Classifier

This project is a **binary image classification system** that predicts whether an image is **AI-generated** or **captured by a human**. It uses **PyTorch** with a fine-tuned **EfficientNet-B0** model and features a **Streamlit web app** for interactive predictions.

---

## 📌 Objective

The primary goal is to detect and distinguish between **synthetic (AI-generated)** and **real (human-taken)** images using deep learning. It also provides an intuitive interface for users to upload images and view prediction results.

---

## 🧰 Libraries Used

| Library         | Purpose                                      |
|-----------------|----------------------------------------------|
| `torch`         | For building and training the model          |
| `torchvision`   | Pretrained EfficientNet and image transforms |
| `Pillow (PIL)`  | Image handling and preprocessing             |
| `streamlit`     | Lightweight web app for predictions          |
| `scikit-learn`  | Precision, Recall, F1-score metrics          |
| `pandas`        | Reading dataset CSVs                         |
| `tqdm`          | Progress bars during training                |

---

## 🏗️ Project Structure

AI_vs_Human_Classifier/ │
├── best_model.pth # Trained model weights
├── app.py # Streamlit web app 
├── predict_single.py # CLI script to predict one image 
├── training_script.py # Full training pipeline 
├── submission.csv # Test predictions 
├── requirements.txt # List of required packages 
├── README.md # Project overview and instructions 
└── venv/ # Optional virtual environment


---

## 🔎 How It Works

### 🖼️ Prediction (Web App - `app.py`)

- Upload an image via the Streamlit interface.
- The image is resized and normalized.
- The model predicts the class and confidence score.
- Output is displayed as: `AI` or `Human`.

### 🖥️ Command-Line Prediction (`predict_single.py`)

python predict_single.py --image_path path_to_your_image.jpg

    Predicts a single image via terminal.

    Returns label and confidence.

💾 Model Details

    Base Model: efficientnet_b0 (pretrained on ImageNet)

    Final Layer: Linear(in_features, 2) for binary output

    Training:

        Only last 3 feature blocks and classifier layers are unfrozen.

        Data augmented with horizontal flip and rotation.

        Early stopping used based on validation accuracy.

    Evaluation:

        Metrics: Precision, Recall, F1-Score on validation set.

🚀 Getting Started
1️⃣ Clone the repository

git clone https://github.com/yourusername/AI_vs_Human_Classifier.git
cd AI_vs_Human_Classifier

2️⃣ Create and activate virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3️⃣ Install the dependencies

pip install -r requirements.txt

4️⃣ Run the web app

streamlit run app.py

5️⃣ (Optional) Predict via CLI

python predict_single.py --image_path path_to_image.jpg

📸 Sample Output

Uploaded Image: ✅
Prediction: AI
Confidence: 98.23%

📊 Performance

    ✅ Val Accuracy: 90%+

    ✅ F1 Score: ~0.92

    ✅ Robustness: Fine-tuned with strong augmentations

    ✅ Hardware Used: AMD Radeon GPU with DirectML or CPU fallback

📦 requirements.txt

torch
torchvision
Pillow
streamlit
scikit-learn
tqdm
pandas

🙌 Acknowledgments

    PyTorch

    Streamlit

    EfficientNet Paper

📬 Contact

For feedback, issues, or contributions, feel free to open an issue or contact:
Your Name – boya saikiran - boyasai@karunya.edu.in

⭐ If you like this project, don't forget to star the repo!


Let me know if you'd like a downloadable `.md` file or want to tailor it for GitHub Pages too.

