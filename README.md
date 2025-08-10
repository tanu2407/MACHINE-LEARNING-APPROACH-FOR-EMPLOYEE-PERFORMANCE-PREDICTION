# MACHINE-LEARNING-APPROACH-FOR-EMPLOYEE-PERFORMANCE-PREDICTION

## 📌 Project Overview
This project predicts **employee productivity** in a garments manufacturing setting based on various operational and work-related factors.  
It uses **Machine Learning** models like:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The best-performing model is deployed via a **Flask web application** with interactive HTML pages.

PROJECT STRUCTURE
📦 Employee-Productivity-Prediction
│
├── app.py # Flask backend application
├── garments_worker_productivity.csv # Dataset
├── gwp.pkl # Saved trained ML model
├── multi_column_label_encoder.pkl # Saved MultiColumnLabelEncoder object
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── static/
│ ├── style.css # CSS styling for HTML pages
│ ├── background.jpg # Background image used in templates
│
├── templates/
│ ├── home.html # Home page
│ ├── about.html # About project page
│ ├── submit.html # Prediction result page
│ ├── predict.html # Input form for prediction
│
├── notebooks/
│ ├── eda_training.ipynb # Jupyter notebook for EDA, preprocessing, model training
│
└── README.md

## 📊 Dataset Information
**Source:** `garments_worker_productivity.csv`  
The dataset contains:
- **Categorical features:** `quarter`, `department`, `day`
- **Numeric features:** `targeted_productivity`, `over_time`, `idle_time`, `no_of_style_change`, `month`, `smv`, `incentive`, `idle_men`, `no_of_workers`
- **Target variable:** `actual_productivity`

---

## 🔍 Machine Learning Workflow
1. **Data Preprocessing**
   - Handling missing values
   - Converting `quarter`, `department`, and `day` to categorical values using a **MultiColumnLabelEncoder**
   - Keeping `month` as a numeric feature
2. **Exploratory Data Analysis (EDA)**
   - Descriptive statistics
   - Correlation analysis
3. **Model Training**
   - Linear Regression
   - Random Forest Regressor
   - XGBoost Regressor
4. **Model Evaluation**
   - Metrics: R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
   - Selecting the best model
5. **Model Saving**
   - Saving the trained model as `gwp.pkl`
   - Saving the encoder as `multi_column_label_encoder.pkl`
6. **Deployment**
   - Flask backend (`app.py`)
   - HTML templates for user interaction

---

## 🌐 Web App Pages
- **Home (`home.html`)**
  - Introductory page with navigation links
- **About (`about.html`)**
  - Information about the project and technology stack
- **Predict (`predict.html`)**
  - Form to input employee details
- **Submit (`submit.html`)**
  - Displays prediction results with productivity interpretation
 
- The app will be accessible at-
http://127.0.0.1:5000/
