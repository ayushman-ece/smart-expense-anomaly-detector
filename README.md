# 💰 Smart Expense Anomaly Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-orange)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Isolation%20Forest-red?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

The Smart Expense Anomaly Detection System is a Python-based Machine Learning project that identifies unusual spending transactions using the Isolation Forest algorithm.

It analyzes transaction amounts, transaction times, days, and expense categories to detect spending patterns that differ from normal transactions.

## 🎯 Objectives

- Analyze personal expense transactions.
- Identify unusual spending patterns.
- Apply unsupervised Machine Learning.
- Detect anomalous transactions using Isolation Forest.
- Visualize normal and unusual expenses.
- Generate an expense anomaly report.

## ⚙️ Features

- CSV dataset loading and validation.
- Category-wise spending analysis.
- Average and total expense calculation.
- Numerical feature scaling.
- Categorical feature encoding.
- Isolation Forest anomaly detection.
- New expense anomaly prediction.
- Anomaly scoring.
- Expense visualization.
- CSV report generation.

## 🧠 Workflow

1. Load and validate expense data.
2. Analyze spending patterns.
3. Scale numerical features.
4. Encode expense categories.
5. Train the Isolation Forest model.
6. Detect unusual transactions.
7. Predict whether a new expense is anomalous.
8. Visualize and export results.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Visualization |
| Isolation Forest | Anomaly detection |

## 📊 Dataset Information

The sample dataset contains 40 synthetic transactions.

| Feature | Description |
|---|---|
| transaction_id | Unique transaction identifier |
| amount | Expense amount |
| hour | Transaction hour (0–23) |
| day_of_week | Day number (1–7) |
| category | Expense category |

## 🔍 Anomaly Detection

Isolation Forest identifies unusual observations by isolating them through random decision trees.

- `1` = Normal transaction
- `-1` = Anomalous transaction

The model uses a contamination setting of 0.15, representing an assumed anomaly proportion of 15% for this experiment.

## 📁 Generated Files

| File | Description |
|---|---|
| expense_anomalies.png | Normal vs anomalous expense chart |
| expense_anomaly_report.csv | Complete anomaly detection report |

## 🚀 Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/ayushman-ece/smart-expense-anomaly-detector.git
```

**2. Open the project folder**

```bash
cd smart-expense-anomaly-detector
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate on Windows**

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the project**

```bash
python main.py
```

## 📚 Key Concepts Learned

- Unsupervised Machine Learning
- Anomaly Detection
- Isolation Forest
- Feature Scaling
- One-Hot Encoding
- Scikit-learn Pipelines
- Decision Function
- Data Visualization
- CSV report generation

## ⚠️ Limitations

- Uses only 40 synthetic transactions.
- Does not learn actual individual spending habits.
- Unusual does not necessarily mean fraudulent.
- Results depend on the contamination setting and dataset.
- Not suitable for real financial fraud detection without further validation.

## 🔮 Future Improvements

- Use real personal expense datasets.
- Add monthly spending pattern analysis.
- Build personalized anomaly thresholds.
- Add transaction history and user profiles.
- Compare Isolation Forest with Local Outlier Factor.
- Develop a Streamlit expense monitoring dashboard.

## 👨‍💻 Author

**Ayushman Tiwari**

GitHub: [@ayushman-ece](https://github.com/ayushman-ece)

---

⭐ If you find this project useful, consider giving the repository a star!