# 🏠 Real Estate Price Prediction Dashboard

A full-stack Machine Learning application built with **Streamlit** that predicts real estate prices, recommends similar properties, and performs market segmentation using clustering.

---

## 🚀 Features

### 💰 Price Prediction
- Predict property prices using trained **ensemble models**
- Models used:
  - Stacking Regressor
  - Voting Regressor

### 🔍 Property Recommendations
- Content-Based Filtering (Cosine Similarity)
- K-Nearest Neighbors (KNN)
- Displays similar properties with scores

### 📊 Market Segmentation
- PCA (Dimensionality Reduction)
- K-Means Clustering
- Cluster visualization + statistics

---

## 📂 Project Structure

```
real-estate-prediction/
├── dashboard/
│   ├── app.py
│   └── README.md
├── src/
│   ├── data_loader.py
│   ├── ensemble.py
│   ├── recommendation.py
│   └── clustering.py
├── requirements.txt
```

---

## ⚠️ Important Note About Models

This project **does NOT store trained models in the repository** (to avoid large file issues).

Instead:
- Models are hosted on **Google Drive**
- They are downloaded automatically when the app runs

### Required Model Files
- `stacking_ensemble.joblib`
- `voting_ensemble.joblib`

### Google Drive Setup
Make sure:
- Files are uploaded to Google Drive  
- Sharing is set to: **Anyone with the link**

---

## 🧰 Prerequisites

- Python 3.9+
- pip
- Git

---

## ⚙️ Setup Instructions (Local)

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd real-estate-prediction
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows
```bash
venv\Scripts\activate
```

#### Mac/Linux
```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

```bash
streamlit run dashboard/app.py
```

---

### 🌐 Access the App

```
http://localhost:8501
```

---

## ☁️ Deployment (Streamlit Community Cloud)

### Step 1: Push Code to GitHub

```bash
git add .
git commit -m "deploy: streamlit app"
git push
```

---

### Step 2: Deploy on Streamlit Cloud

Go to:  
👉 https://share.streamlit.io  

Fill in:

| Field | Value |
|------|------|
| Repository | your-username/repo-name |
| Branch | main |
| Main file path | dashboard/app.py |

Click **Deploy**

---

### 🌍 Live App URL

```
https://your-app-name.streamlit.app
```

---

## ⚠️ Deployment Notes

- Models are downloaded dynamically from Google Drive  
- No need to commit `.joblib` files  
- First app load may take **20–40 seconds**

---

## 📦 requirements.txt

Ensure your `requirements.txt` contains:

```
streamlit
numpy
pandas
scikit-learn
matplotlib
gdown
joblib
```

---

## ❗ Common Issues & Fixes

### ❌ Models not loading
- Ensure Google Drive files are public  
- Check file names:
  - `stacking_ensemble.joblib`
  - `voting_ensemble.joblib`

---

### ❌ App crashes on deployment
- Check missing dependencies in `requirements.txt`

---

### ❌ Slow loading
- Normal for first run (models download)

---

## 🎯 Final Outcome

- ✔ End-to-end ML application  
- ✔ No large files in GitHub  
- ✔ Production-ready deployment  
- ✔ Interactive UI for predictions and analytics  

---

## 👨‍💻 Author

**Nagasantosh Chandrashekar Chavvakula**