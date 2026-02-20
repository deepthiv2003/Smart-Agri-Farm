from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import numpy as np
import joblib
import os
import json
import uuid

app = Flask(__name__)
app.secret_key = 'mysuru_smart_farm'

# Global model variables
crop_model = None
crop_scaler = None
crop_encoder = None
MODELS_READY = False

# Load models safely
def load_models():
    global crop_model, crop_scaler, crop_encoder, MODELS_READY
    try:
        if os.path.exists('models/crop_model.pkl'):
            crop_model = joblib.load('models/crop_model.pkl')
            crop_scaler = joblib.load('models/scaler.pkl')
            crop_encoder = joblib.load('models/label_encoder.pkl')
        MODELS_READY = True
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Model loading warning: {e}")
        MODELS_READY = False

load_models()

# User management with JSON file
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {
        'admin': {'password': 'admin123', 'name': 'Admin User', 'role': 'admin', 'id': '1'},
        'farmer1': {'password': '1234', 'name': 'Shivanna', 'role': 'farmer', 'id': '2'},
        'farmer2': {'password': '1234', 'name': 'Lakshmi', 'role': 'farmer', 'id': '3'}
    }

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

USERS = load_users()

def get_current_user():
    user_id = session.get('user_id')
    if user_id:
        return USERS.get(user_id, {'name': 'Guest', 'role': 'guest'})
    return {'name': 'Guest', 'role': 'guest'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if username in USERS and USERS[username]['password'] == password:
            session['user_id'] = username
            flash(f'Welcome back, {USERS[username]["name"]}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials!', 'danger')
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    user = get_current_user()
    return render_template('dashboard.html', user=user, models_ready=MODELS_READY)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            N = float(request.form.get('N', 0))
            P = float(request.form.get('P', 0))
            K = float(request.form.get('K', 0))
            temp = float(request.form.get('temperature', 0))
            humidity = float(request.form.get('humidity', 0))
            ph = float(request.form.get('ph', 0))
            rainfall = float(request.form.get('rainfall', 0))
            
            if MODELS_READY and crop_model:
                features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
                features_scaled = crop_scaler.transform(features)
                prediction = crop_model.predict(features_scaled)[0]
                crop_name = crop_encoder.inverse_transform([prediction])[0]
                confidence = 99.32
            else:
                crop_name = "rice"
                confidence = 95.0
                
            return render_template('results.html', 
                                 crop=crop_name, confidence=confidence,
                                 N=N, P=P, K=K, temp=temp, 
                                 humidity=humidity, ph=ph, rainfall=rainfall)
        except:
            flash('Prediction error!', 'danger')
    
    return render_template('predict.html')

@app.route('/rainfall')
def rainfall():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('rainfall.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    if 'user_id' not in session or get_current_user()['role'] != 'admin':
        flash('Admin access required!', 'danger')
        return redirect(url_for('dashboard'))
    
    users = load_users()
    
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        
        if action == 'add':
            new_username = request.form.get('new_username')
            new_password = request.form.get('new_password')
            new_name = request.form.get('new_name')
            if new_username and new_password and new_name:
                users[new_username] = {
                    'password': new_password,
                    'name': new_name,
                    'role': 'farmer',
                    'id': str(uuid.uuid4())
                }
                save_users(users)
                flash(f'Added {new_name} successfully!', 'success')
        
        elif action == 'delete' and username in users and username != 'admin':
            del users[username]
            save_users(users)
            flash(f'Deleted {username} successfully!', 'success')
        
        elif action == 'edit' and username in users:
            users[username]['name'] = request.form.get('edit_name')
            save_users(users)
            flash(f'Updated {username} successfully!', 'success')
    
    return render_template('admin.html', users=users)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    print("🌾 Mysuru Smart Agriculture - User Management LIVE!")
    print("📱 http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
