from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# Custom encoder definition
# -------------------------
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        from sklearn.preprocessing import LabelEncoder
        self.encoders = {}
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        output = X.copy()
        for col in self.columns:
            if col in output.columns:
                output[col] = self.encoders[col].transform(output[col])
        return output

# -------------------------
# Flask app init
# -------------------------
app = Flask(__name__)

# Load trained model & encoder
model = joblib.load("gwp.pkl")
label_encoders = joblib.load("multi_column_label_encoder.pkl")

# Expected feature order (month stays numeric)
feature_order = [
    'quarter', 'day', 'targeted_productivity', 'over_time', 'idle_time',
    'no_of_style_change', 'month', 'department', 'team', 'smv',
    'incentive', 'idle_men', 'no_of_workers'
]

# Define categorical columns for encoding
categorical_cols = ['quarter', 'department', 'day']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/submit')
def submit():
    return render_template('submit.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect form inputs in correct order
            input_data = []
            for feature in feature_order:
                input_data.append(request.form[feature])

            # Create DataFrame
            input_df = pd.DataFrame([input_data], columns=feature_order)

            # Convert numeric columns (month is numeric, not categorical)
            numeric_cols = [
                'targeted_productivity', 'over_time', 'idle_time',
                'no_of_style_change', 'month', 'smv', 'incentive',
                'idle_men', 'no_of_workers'
            ]
            for col in numeric_cols:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

            # Apply label encoding to categorical columns only
            input_df = label_encoders.transform(input_df)

            # Predict
            prediction = model.predict(input_df)[0]

            # Productivity logic
            if prediction >= 0.8:
                result_text = f"Highly productive({prediction:.2f})"
            elif prediction >= 0.5:
                result_text = f"Medium productive ({prediction:.2f})"
            else:
                result_text = f"Less productive ({prediction:.2f})"

            return render_template('submit.html', prediction_text=result_text)

        except Exception as e:
            return f"Error: {e}"
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)
















