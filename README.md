# 📈 MarketMinds – AI Smart Stock Advisor

## 📌 Project Overview
MarketMinds is a Python-based AI-powered stock market analysis application designed to help users understand Indian stock market trends through data analysis and Machine Learning.

The application analyzes historical stock market data, calculates technical indicators, and uses a Linear Regression model to predict the next trading day's closing price for selected Indian stocks.

## 🎯 Objectives
- Analyze historical Indian stock market data.
- Perform data cleaning and preprocessing.
- Calculate technical indicators for stock analysis.
- Apply Machine Learning for stock price prediction.
- Display stock insights through an interactive web application.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Stock Market Data:** Yahoo Finance (yfinance)
- **Frontend:** Streamlit
- **Model Storage:** Joblib

## 📊 Features

- Historical stock market data collection.
- Data cleaning and preprocessing.
- Technical indicator calculations:
  - Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Volume Change
- Linear Regression-based stock price prediction.
- Interactive Streamlit dashboard.
- Stock selection and visualization.
- Display of stock analysis and predicted prices.

## 📈 Supported Indian Stocks

The application currently supports:

- RELIANCE – Reliance Industries
- TCS – Tata Consultancy Services
- INFY – Infosys
- ICICIBANK – ICICI Bank
- M&M – Mahindra & Mahindra

## 🧠 Machine Learning Model

MarketMinds uses **Linear Regression**, a supervised Machine Learning algorithm, to estimate the next trading day's closing price.

**Model Development Steps:**
1. Collect historical stock market data.
2. Clean and preprocess the dataset.
3. Perform feature engineering.
4. Calculate technical indicators.
5. Create the prediction target using the next trading day's closing price.
6. Split the dataset into training and testing sets.
7. Train the Linear Regression model.
8. Evaluate model performance using MAE, RMSE and R² score.
9. Save the trained model using Joblib.

## 📂 Project Structure

```text
MarketMinds/
│
├── app.py
├── data_fetch.py
├── data_analysis.py
├── model.py
├── requirements.txt
├── marketminds_model.joblib
│
└── data/
    └── RELIANCE_cleaned.csv
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MarketMinds.git
```

### 2. Navigate to the Project Folder

```bash
cd MarketMinds
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📚 Learning Outcomes

- Practical understanding of Machine Learning concepts.
- Experience with real-world financial datasets.
- Data cleaning, preprocessing and feature engineering.
- Implementation and evaluation of Linear Regression.
- Building interactive applications using Streamlit.
- Integrating Machine Learning models with a Python frontend.

## ⚠️ Disclaimer

MarketMinds is an educational Machine Learning project developed for learning and demonstration purposes. Stock price predictions are estimates and should not be considered financial or investment advice.

## 👨‍💻 Developer

**Mohammed Mahadhir**

B.Sc Information Technology

Interests: Python | Data Science | Machine Learning | Artificial Intelligence
