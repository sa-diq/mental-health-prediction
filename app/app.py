from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join('models', 'mental_health_classifier_v1.joblib')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    features = {
        'Age': float(request.form['age']),
        'Gender': request.form['gender'],
        'self_employed': request.form['self_employed'],
        'family_history': request.form['family_history'],
        'work_interfere': request.form['work_interfere'],
        'no_employees': request.form['no_employees'],
        'remote_work': request.form['remote_work'],
        'tech_company': request.form['tech_company'],
        'benefits': request.form['benefits'],
        'care_options': request.form['care_options'],
        'wellness_program': request.form['wellness_program'],
        'seek_help': request.form['seek_help'],
        'anonymity': request.form['anonymity'],
        'leave': request.form['leave'],
        'mental_health_consequence': request.form['mental_health_consequence'],
        'phys_health_consequence': request.form['phys_health_consequence'],
        'coworkers': request.form['coworkers'],
        'supervisor': request.form['supervisor'],
        'mental_health_interview': request.form['mental_health_interview'],
        'phys_health_interview': request.form['phys_health_interview'],
        'mental_vs_physical': request.form['mental_vs_physical'],
        'obs_consequence': request.form['obs_consequence']
    }
    
    # Create DataFrame from features
    df = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return render_template('result.html', 
                         prediction=prediction,
                         probability=round(probability * 100, 2))

if __name__ == '__main__':
    app.run(debug=True) 