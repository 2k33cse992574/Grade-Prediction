# AI-Powered Student Grade Predictor

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-blue?style=for-the-badge)

**Student Grade Predictor** is a machine learning web application that predicts a student's final grade category (A or B) based on several key performance indicators. It demonstrates the application of predictive analytics in education to help identify students who may need additional academic support.

## ✨ Features

- **Predictive Modeling:** Uses a trained classifier to forecast grades (A or B).
- **Key Performance Features:**
  - **Weekly Self-Study Hours:** Tracks time investment outside of class.
  - **Attendance Percentage:** Measures student consistency and presence.
  - **Class Participation:** Captures engagement level during lectures.
- **Probability Scores:** Provides the underlying probability of the prediction for more nuanced insights.
- **RESTful API:** Flask-based backend for serving real-time predictions.

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Flask-CORS.
- **Data Handling:** Pandas, NumPy.
- **Machine Learning:** Scikit-Learn.
- **Serialization:** Pickle (for model and scaler artifacts).

## 🚀 Getting Started

### Prerequisites
- Python 3.x

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/2k33cse992574/Grade-Prediction.git
   cd Grade-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```
   *The application will launch on `http://localhost:5000`.*

## 📝 License
This project is open-source.
