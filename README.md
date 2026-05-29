# AI-Powered Student Performance Prediction System

**MCA 4th Semester Major Project | Chandigarh University**  
**Subject Code:** 23ONMCR-753  
**Student:** Vinay Kumar Rajput | Enrollment No.: 024MCA110508

---

## Overview

A web-based application that predicts student academic performance (At-Risk / Average / High Performer) using supervised Machine Learning algorithms. Built with Python, Flask, Scikit-learn, and MySQL.

**Best Model:** Random Forest Classifier — **87.5% accuracy**, weighted F1-Score: 0.874

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 2.3 |
| ML | Scikit-learn 1.3, Pandas, NumPy, Joblib |
| Database | MySQL 8.0, Flask-SQLAlchemy |
| Frontend | HTML5, Bootstrap 5, Chart.js |
| Auth | Flask-Login, Flask-Bcrypt |

---

## Project Structure

```
student_prediction_system/
├── app/
│   ├── __init__.py          # App factory, DB init
│   ├── models.py            # SQLAlchemy models
│   ├── routes.py            # Flask routes
│   ├── ml_module.py         # ML prediction engine
│   ├── templates/
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── dashboard.html
│   │   ├── students.html
│   │   ├── predict.html
│   │   └── result.html
│   └── static/
│       └── css/style.css
├── models/                  # Serialized .pkl model files (gitignored)
├── scripts/
│   └── create_db.sql        # MySQL schema
├── tests/
│   ├── test_ml.py
│   └── test_routes.py
├── train_model.py           # ML training script
├── seed_users.py            # Admin/faculty seed script
├── run.py                   # App entry point
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- MySQL 8.0 Community Edition

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/student_prediction_system.git
cd student_prediction_system

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the MySQL database
mysql -u root -p < scripts/create_db.sql

# 5. Update DB credentials in app/__init__.py
# Change: SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:PASSWORD@localhost/spp_db'

# 6. Train the ML model (generates .pkl files in models/ folder)
python train_model.py

# 7. Seed admin and faculty users
python seed_users.py

# 8. Run the application
python run.py
```

Open your browser at: **http://localhost:5000/login**

---

## ML Algorithms Compared

| Algorithm | Accuracy | Weighted F1 |
|-----------|----------|-------------|
| Logistic Regression | 81.2% | 0.811 |
| Decision Tree (C4.5) | 83.8% | 0.836 |
| SVM (RBF Kernel) | 85.1% | 0.850 |
| **Random Forest ✓** | **87.5%** | **0.874** |

---

## Input Features

| Feature | Type | Range |
|---------|------|-------|
| Attendance Percentage | Numeric | 0–100 |
| Internal Assessment Marks | Numeric | 0–100 |
| Assignment Score | Numeric | 0–100 |
| Previous Semester GPA | Numeric | 0–10 |
| Study Hours Per Day | Numeric | 0–12 |
| Class Participation Score | Numeric | 0–10 |
| Gender | Categorical | Male/Female |
| Family Income Level | Categorical | Low/Medium/High |
| Extra-Curricular Participation | Binary | 0/1 |

---

## License

Academic project submitted to Chandigarh University. Not licensed for commercial use.
