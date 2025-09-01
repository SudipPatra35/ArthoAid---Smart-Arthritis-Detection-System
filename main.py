import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for
import pymysql
import os
from werkzeug.utils import secure_filename
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TF logs
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# import all the models 
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
cnn = load_model("model/best_model_1.h5")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for sessions and flash messages


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MySQL Database Connection
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="Sudip@7797",
        database="arthoaid",
        port=3306
    )

# --- HOME ROUTE with form toggle ---
@app.route('/')
def home():
    form_type = request.args.get('form', 'login')  # 'login' by default
    return render_template('login.html', form_type=form_type)


# --- LOGIN ROUTE ---
@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()

    if user and user[2] == password:  # Assuming password is 3rd column
        session['email'] = user[0]  # Assuming email is 1st column
        flash('User logged in successfully!', 'success')
        return redirect(url_for('dashboard'))

    flash('Invalid Email or Password', 'danger')
    return redirect(url_for('home', form='login'))


# --- REGISTER ROUTE ---
@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if password != confirm_password:
        flash('Passwords do not match', 'danger')
        return redirect(url_for('home', form='register'))

    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
        if cursor.fetchone():
            flash('Email already registered', 'warning')
            return redirect(url_for('home', form='register'))

        cursor.execute("INSERT INTO user (email, name, password) VALUES (%s, %s, %s)",
                       (email, name, password))
        connection.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('home', form='login'))
    except Exception as e:
        flash(f'Registration failed: {str(e)}', 'danger')
        return redirect(url_for('home', form='register'))
    finally:
        cursor.close()
        connection.close()


# --- DASHBOARD ROUTE ---
@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('home', form='login'))
    email = session['email']
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    if user:
        return render_template('dashboard.html', user=user)
    else:
        flash('User not found', 'danger')
        return redirect(url_for('home', form='login'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- Extract numeric inputs ---
        age = int(request.form['age'])
        rf = float(request.form['rf'])
        anti_ccp = float(request.form['anti_ccp'])
        esr = float(request.form['esr'])
        crp = float(request.form['crp'])
        joint_pain = int(request.form['joint_pain'])
        stiffness = int(request.form['stiffness'])

        # --- Extract categorical inputs ---
        sex = int(request.form['sex'])         # 1 = Male, 0 = Female
        swelling = int(request.form['swelling'])  # 1 = Yes, 0 = No
        

         # --- Create list of input values ---
        numeric_input = np.array([[age, rf, anti_ccp, esr, crp, joint_pain, stiffness]])

        categorical_input = np.array([[sex, swelling]])

        scaler_numeric = scaler.transform(numeric_input)

        final_input = np.hstack((scaler_numeric, categorical_input))

        prediction = model.predict(final_input)[0]
        # print(f"Prediction result: {prediction}")  # Debug log
       
        # --- Flash message (optional) ---
        flash("Data received successfully!", "success")
        
        # --- Render dashboard with result ---
        return render_template('dashboard.html', input_data=prediction)

    except Exception as e:
        flash(f"Error processing input: {str(e)}", "danger")
        return redirect(url_for('dashboard'))

    
@app.route('/upload_xray', methods=['POST'])
def upload_xray():
    
    try:
        has_xray = int(request.form['has_xray'])

        if has_xray == 1 :
            xray_image = request.files['xray_image']

            if xray_image.filename != '':
                # print(has_xray)
                filename = secure_filename(xray_image.filename)
                xray_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                xray_image.save(xray_path)
                flash('X-ray uploaded successfully!', 'success')
                
                # Predict
                # print(xray_path)
                img_array = preprocess_image(xray_path)
                pred = cnn.predict(img_array)
                predicted_class = np.argmax(pred)
                class_labels = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
                print("Predicted class:", class_labels[predicted_class])
                return render_template('dashboard.html', result=class_labels[predicted_class])
            else:
                flash('No file selected.', 'danger')
        else:
            flash('X-ray not uploaded or not required.', 'warning')
    except Exception as e:
        flash(f'Error uploading X-ray: {str(e)}', 'danger')

    return redirect(url_for('dashboard'))


def preprocess_image(path):
    from PIL import Image
    img = Image.open(path).convert("L")  # Convert to grayscale
    img = img.resize((256, 256))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return img_array

# --- LOGOUT ROUTE (optional) ---
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home', form='login'))

# --- RUN APP ---
if __name__ == '__main__':
    app.run(debug=True)
