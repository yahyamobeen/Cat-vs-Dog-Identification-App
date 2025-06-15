```markdown
🐱🐶 Cat vs Dog Classifier

An end-to-end image classification project using Convolutional Neural Networks (CNNs) to distinguish between cats and dogs. This project covers data preprocessing, model training, evaluation, and interactive deployment using Streamlit.

---

## 🗂️ Project Structure

```

cat-vs-dog-project/
├── data/
│   ├── train/         # Training images
│   │   ├── cat/
│   │   └── dog/
│   └── test/          # Testing images
│       ├── cat/
│       └── dog/
├── notebooks/         # EDA notebooks (Jupyter)
├── results/           # Evaluation outputs (confusion matrix, reports)
├── model/             # Trained model (cat\_dog\_model.h5)
├── app/               # Streamlit app
├── preprocess.py      # Preprocessing & data loader
├── train\_model.py     # CNN training script
├── evaluate\_model.py  # Model evaluation script
└── README.md          # Project documentation

````

---

## 🔧 Setup Instructions

### 1. ✅ Install Requirements

Make sure you’re using **Python 3.10+** (Python 3.11 supported with `tensorflow-cpu`):

```bash
pip install -r requirements.txt
````

---

### 2. 🧹 Clean the Dataset (Optional)

Run this script to delete any corrupted images (recommended before training):

```bash
python clean_dataset.py
```

---

### 3. 🧠 Train the Model

```bash
python train_model.py
```

This trains a CNN and saves the model as `model/cat_dog_model.h5`.

---

### 4. 📊 Evaluate Model Performance

```bash
python evaluate_model.py
```

This will:

* Generate a **confusion matrix**: `results/confusion_matrix.png`
* Save a **classification report**: `results/classification_report.txt`

---

### 5. 🌐 Run Streamlit Web App

```bash
python -m streamlit run app/streamlit_app.py
```

> Use `python -m` if `streamlit` is not recognized directly.

---

## 🧪 Model Info

* **Architecture**: 2 Convolutional Layers + Dense Layer
* **Loss**: Binary Crossentropy
* **Metrics**: Accuracy, F1-score
* **Input Shape**: 128x128x3 (RGB)

---

## 📌 Course Alignment

This project covers:

* ✅ Exploratory Data Analysis (EDA)
* ✅ Data Preprocessing
* ✅ Model Training & Evaluation
* ✅ Interactive App with Streamlit

---

## 📬 Author

**Yahya Mobeen**
Introduction to Data Science – Summer 2025


