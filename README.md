# 🏠 Rental & Housing Price Prediction

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Visualization](https://img.shields.io/badge/Visualization-Plotly-blueviolet.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application designed to predict real estate and rental property prices based on housing attributes, structural characteristics, amenities, and location factors. Powered by a **Random Forest Regressor** and presented through an interactive **Streamlit** dashboard with dynamic **Plotly** data visualisations.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset Summary](#-dataset-summary)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Interactive Visualizations](#-interactive-visualizations)
- [Tech Stack](#-tech-stack)
- [Authors & Acknowledgments](#-authors--acknowledgments)

---

## 📖 Overview

Real estate pricing and rental valuation depend heavily on multiple structural, environmental, and amenity variables. This project provides a robust, data-driven approach to estimating property prices accurately.

Using historical housing transaction records, the model analyzes relationships between area size, room configurations, air conditioning, basement availability, parking, and locality to produce instantaneous valuations. The interactive frontend allows users to tweak property parameters on the fly and compare predicted values against overall market distributions.

---

## ✨ Key Features

- **⚡ Real-Time Price Estimation**: Instant price calculation (Yearly and estimated Monthly breakdown) as input parameters are configured.
- **🎛️ Dynamic Sidebar Controls**: User-friendly sliders, number pickers, and select boxes for configuring:
  - Property Area (sq ft)
  - Bedroom & Bathroom counts
  - Number of Stories / Floors
  - Parking slot capacity
  - Furnishing status (Furnished, Semi-Furnished, Unfurnished)
  - Amenities (Air conditioning, Guest room, Basement, Hot water heating, Main road access, Prime locality)
  - Regional focus for Tamil Nadu and Puducherry districts.
- **📊 Interactive Analytics Dashboard**:
  - **Price Distribution with Value Marker**: Interactive histogram demonstrating market price distribution with an overlay line for the current prediction.
  - **Feature Correlation Heatmap**: Annotated correlation matrix uncovering dependencies between quantitative housing features.
- **🚀 Caching & High Performance**: Powered by Streamlit's `@st.cache_resource` and `@st.cache_data` for rapid page rendering.

---

## 📂 Dataset Summary

The dataset (`data/Housing.csv`) contains 545 observations across 13 property features:

| Feature | Type | Description |
|---|---|---|
| `price` | Numerical | Target variable: property price / valuation in INR (₹) |
| `area` | Numerical | Total surface area of the property in square feet |
| `bedrooms` | Numerical | Total count of bedrooms (1 - 6) |
| `bathrooms` | Numerical | Total count of bathrooms (1 - 4) |
| `stories` | Numerical | Number of stories/floors in the structure (1 - 4) |
| `mainroad` | Categorical | Proximity / connectivity to main road (`yes` / `no`) |
| `guestroom` | Categorical | Availability of a dedicated guest room (`yes` / `no`) |
| `basement` | Categorical | Presence of a basement (`yes` / `no`) |
| `hotwaterheating` | Categorical | Availability of hot water heating facility (`yes` / `no`) |
| `airconditioning` | Categorical | Availability of central air conditioning (`yes` / `no`) |
| `parking` | Numerical | Number of private car parking spaces (0 - 3) |
| `prefarea` | Categorical | Whether located in a prime / preferred residential neighborhood (`yes` / `no`) |
| `furnishingstatus` | Categorical | Furnishing condition (`furnished`, `semi-furnished`, `unfurnished`) |

---

## 🧠 Machine Learning Pipeline

```
  ┌─────────────────┐
  │ data/Housing.csv│
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ Preprocessing & One-Hot │
  │        Encoding         │
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ Train / Test Split      │
  │ (80% Train, 20% Test)   │
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ RandomForestRegressor   │
  │ (n_estimators=200)      │
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ Model Serialization     │
  │      (model.pkl)        │
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ Streamlit App Inference │
  │        (app.py)         │
  └─────────────────────────┘
```

1. **Preprocessing & Encoding**:
   - Categorical string variables are converted into binary flags using one-hot encoding (`pd.get_dummies(..., drop_first=True)`).
2. **Train / Test Split**:
   - 80% training data and 20% validation split (`random_state=42`) to measure generalizability.
3. **Model Selection**:
   - **Random Forest Regressor** with 200 estimators (`n_estimators=200`). Non-linear tree ensembling handles complex feature interactions effectively and mitigates overfitting.
4. **Serialization**:
   - Saved via `pickle` as `model.pkl` for immediate inference in the production web app.

---

## 📁 Project Structure

```
Rental-price-prediction/
├── data/
│   └── Housing.csv          # Housing dataset used for training & analysis
├── app.py                   # Interactive Streamlit web application
├── model.py                 # ML pipeline: data preprocessing, training & export
├── model.pkl                # Serialized pre-trained Random Forest model
├── requirements.txt         # Python project dependencies
├── .gitignore               # Ignored files and directories for Git
└── README.md                # Comprehensive documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed on your system.
- Git installed.

### 1. Clone the Repository
```bash
git clone https://github.com/Devakumar-A/Rental-price-prediction.git
cd Rental-price-prediction
```

### 2. Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Launch the Streamlit App
```bash
streamlit run app.py
```
After executing, Streamlit will start a local development server and automatically open the application in your browser (default: `http://localhost:8501`).

### (Optional) Retrain the Model
To re-train the Random Forest model from scratch and update `model.pkl`:
```bash
python model.py
```

---

## 📈 Interactive Visualizations

The dashboard contains dedicated exploratory views:
- **Price Distribution Analysis**: Visualizes the dispersion of property values across the dataset with skewness indicators and a real-time comparison line for the user's predicted price.
- **Correlation Heatmap**: Unveils positive and negative relationships between physical house dimensions (area, bedrooms, bathrooms) and listing prices.

---

## 💻 Tech Stack

- **Core**: Python 3
- **Web App**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/)
- **Data Manipulation**: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/)
- **Data Visualization**: [Plotly](https://plotly.com/)

---

## 👥 Authors & Acknowledgments

- **Developer**: Devakumar A ([@Devakumar-A](https://github.com/Devakumar-A))
- **Team**: RGCET TECHNOCRATS
