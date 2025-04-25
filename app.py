from flask import Flask, render_template, Response, redirect, url_for
from flask_login import LoginManager, UserMixin, login_required, current_user, login_user, logout_user
import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, flash
from flask_mail import Mail, Message
import firebase_admin
from firebase_admin import credentials, firestore
import csv
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize Firebase
cred = credentials.Certificate("./login-dec57-firebase-adminsdk-fbsvc-c4e0efa37c.json") 
firebase_admin.initialize_app(cred)
db = firestore.client()

# User model (with name)
class User(UserMixin):
    def __init__(self, id, email, name):  # Add name
        self.id = id
        self.email = email
        self.name = name  # Add name

# User loader (load name)
@login_manager.user_loader
def load_user(user_id):
    user_ref = db.collection('users').document(user_id)
    user_data = user_ref.get()
    if user_data.exists:
        user_dict = user_data.to_dict()
        return User(user_id, user_dict.get('email'), user_dict.get('name'))  # add name
    return None

# -------------------------------------------------Authentication routes---------------------------------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'] 
        email = request.form['email']
        password = request.form['password']
        
        # Check if user exists
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).limit(1).get()
        
        if len(query) > 0:
            flash('Email already exists')
            return redirect(url_for('register'))
        
        # Create new user
        user_id = users_ref.document().id
        users_ref.document(user_id).set({
            'name': name,  
            'email': email,
            'password': generate_password_hash(password),
            'created_at': datetime.now()
        })
        
        # Log the user in
        user = User(user_id, email, name) 
        login_user(user)
        flash('Registration successful!')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Find user by email
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).limit(1).get()
        
        if len(query) == 0:
            flash('Invalid email or password')
            return redirect(url_for('login'))
        
        user_data = query[0].to_dict()
        if not check_password_hash(user_data['password'], password):
            flash('Invalid email or password')
            return redirect(url_for('login'))
        
        # Log the user in
        user = User(query[0].id, email, user_data.get('name')) 
        login_user(user)
        flash('Login successful!', 'success') 

        # Clear the user's detection CSV file on login
        clear_user_detection_csv(current_user.id)
        return redirect(url_for('home'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logout successful!', 'success')
    return redirect(url_for('login'))

# Function to clear the user's detection CSV file
def clear_user_detection_csv(user_id):
    user_csv = f"user_{user_id}_detections.csv"
    if os.path.exists(user_csv):
        try:
            with open(user_csv, 'w', newline='') as f:
                pass  # Open in write mode to clear the file
            print(f"Cleared detection data for user {user_id} in {user_csv}")
        except Exception as e:
            print(f"Error clearing detection data for user {user_id}: {e}")


# Load trained models
emotion_model = tf.keras.models.load_model("./Model/emotion_model.keras")
gender_model = tf.keras.models.load_model("./Model/gender-original-retrain.keras")

# Define emotion and gender labels
emotion_labels = ["positive", "negative", "neutral"]
gender_labels = ["Male", "Female"]

# Initialize webcam (moved inside a function)
cap = None

# Store detected data
detection_data = []

# Keep track of detected faces to avoid duplicates
detected_face_regions = []
min_face_distance = 50  # Minimum distance between faces to consider them different
cooldown_time = 50  # Seconds to wait before considering the same face region as a new detection

def generate_frames():
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load OpenCV face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Check if this face is new or already detected
            current_time = time.time()
            face_center = (x + w // 2, y + h // 2)
            is_new_face = True

            # Check against previously detected faces
            for i, (prev_center, timestamp) in enumerate(detected_face_regions):
                # Calculate distance between face centers
                distance = np.sqrt((face_center[0] - prev_center[0]) ** 2 + (face_center[1] - prev_center[1]) ** 2)

                # If face is close to a previously detected one and cooldown hasn't passed
                if distance < min_face_distance and (current_time - timestamp) < cooldown_time:
                    is_new_face = False
                    break
                # If cooldown has passed, remove the old face entry
                elif (current_time - timestamp) >= cooldown_time:
                    detected_face_regions.pop(i)

            # Draw rectangle for all faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Only process new faces
            if is_new_face:
                # Add this face to detected regions
                detected_face_regions.append((face_center, current_time))

                face = gray[y:y + h, x:x + w]

                # Resize to model input size for emotion model
                emotion_face = cv2.resize(face, (48, 48))
                emotion_face = np.expand_dims(emotion_face, axis=-1)  #  channel dimension
                emotion_face = np.expand_dims(emotion_face, axis=0)  #  batch dimension
                emotion_face = emotion_face / 255.0  # Normalize

                # Resize to model input size for gender model
                gender_face = cv2.resize(face, (100, 100))
                gender_face = np.expand_dims(gender_face, axis=-1)  #  channel dimension
                gender_face = np.expand_dims(gender_face, axis=0)  #  batch dimension
                gender_face = gender_face / 255.0  # Normalize

                # Predict emotion
                emotion_pred = emotion_model.predict(emotion_face)
                emotion_index = np.argmax(emotion_pred)
                detected_emotion = emotion_labels[emotion_index]

                # Predict gender
                gender_pred = gender_model.predict(gender_face)
                gender_index = np.argmax(gender_pred)
                detected_gender = gender_labels[gender_index]

                # Store results
                detection_data.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                       detected_emotion, detected_gender])

                # Display results on screen for new faces (with different color)
                label = f"NEW: {detected_gender}, {detected_emotion}"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Add delay to display results for a longer time
                # cv2.imshow("Emotion & Gender Detection", frame)
                #cv2.waitKey(700)  # Delay for 2000 milliseconds (2 seconds)

                print(f"New face detected: {detected_gender}, {detected_emotion}")
            else:
                # Just label existing faces differently
                cv2.putText(frame, "Already detected", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show webcam feed with count of unique faces
        cv2.putText(frame, f"Unique faces: {len(detection_data)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -----------------------------------------------------------Welcome-------------------------------------------------------

@app.route('/')
def welcome():
    return render_template('Welcome.html')

# ------------------------------------------------------------Home---------------------------------------------------------

@app.route('/home')
@login_required
def home():
    return render_template('Home.html')

# -----------------------------------------------------------User Account-------------------------------------------------

@app.route('/user_account')
@login_required
def user_account():
    return render_template('UserAccount.html', current_user=current_user)

# -----------------------------------------------------------capture Start-------------------------------------------------

@app.route('/capture')
@login_required
def capture():
    return render_template('CaptureStart.html')

# ----------------------------------------------------------start_video----------------------------------------------------

@app.route('/start_video')
@login_required
def start_video():
    global cap
    cap = cv2.VideoCapture(1)  # Use 0 for default camera
    if not cap.isOpened():
        flash("Error: Could not open video device")
        return redirect(url_for('capture'))
    return redirect(url_for('capture'))

# -----------------------------------------------------------video_feed----------------------------------------------------

@app.route('/video_feed')
@login_required
def video_feed():
    if cap is None or not cap.isOpened():
        return "Video capture not started."
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------------------------------------------stop_video---------------------------------------------------

@app.route('/stop')
@login_required
def stop():
    global cap, detection_data
    
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    if detection_data and current_user.is_authenticated:
        # Save to local CSV (current feedback)
        user_csv = f"user_{current_user.id}_detections.csv"
        pd.DataFrame(detection_data, columns=["Timestamp", "Emotion", "Gender"]).to_csv(user_csv, index=False)
        
        # Save to Firestore
        user_detections_ref = db.collection('users').document(current_user.id).collection('detections')
        for data in detection_data:
            user_detections_ref.add({
                'timestamp': data[0],
                'emotion': data[1],
                'gender': data[2],
                'user_id': current_user.id
            })
        
        print(f"Saved {len(detection_data)} detections")
        flash('Detections saved successfully!', 'success')
    
    detection_data = []
    
    return redirect(url_for('capture'))

# -----------------------------------------------------------About----------------------------------------------------------

@app.route('/about')
def about():
    return render_template('About.html')

# -----------------------------------------------------------Results------------------------------------------------------

@app.route('/results')
def results():
    return render_template('Results.html')

# ------------------------------------------------------EmotionResults (pie chart)-----------------------------------------

@app.route("/EmotionResults")
@login_required
def EmotionResults():
    try:
    # Load CSV data
        df = pd.read_csv(f"user_{current_user.id}_detections.csv")  # CSV file
    
    except pd.errors.EmptyDataError:
        flash("No current data available for Emotion Analysis. Please ensure customer feedback data is collected.", "warning")
        return render_template("EmotionResults.html", labels=[], positive_values=[], negative_values=[], neutral_values=[], heatmap_labels=[], heatmap_datasets=[])
    
    except FileNotFoundError:
        flash("No current data available for Emotion Analysis. Please ensure customer feedback data is collected.", "warning")
        return render_template("EmotionResults.html", labels=[], positive_values=[], negative_values=[], neutral_values=[], heatmap_labels=[], heatmap_datasets=[])

    # Ensure the columns exist in the dataframe

    if "Timestamp" in df.columns and "Emotion" in df.columns and "Gender" in df.columns:
        labels = df["Timestamp"].tolist()
        emotions = df["Emotion"].tolist()
        genders = df["Gender"].tolist()

        # Calculate emotion counts
        positive_count = emotions.count('positive')
        negative_count = emotions.count('negative')
        neutral_count = emotions.count('neutral')

        # Calculate total count and percentages
        total_count = positive_count + negative_count + neutral_count
        if total_count > 0:
            positive_percentage = round((positive_count / total_count) * 100, 2)
            negative_percentage = round((negative_count / total_count) * 100, 2)
            neutral_percentage = round((neutral_count / total_count) * 100, 2)
        else:
            positive_percentage = 0
            negative_percentage = 0
            neutral_percentage = 0
    else:
        labels = []
        emotions = []
        genders = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        positive_percentage = 0
        negative_percentage = 0
        neutral_percentage = 0

    return render_template("EmotionResults.html", labels=labels, emotions=emotions, genders=genders,
                           positive_count=positive_count, negative_count=negative_count, neutral_count=neutral_count,
                           total_count=total_count,
                           positive_percentage=positive_percentage, negative_percentage=negative_percentage, neutral_percentage=neutral_percentage)

# ---------------------------------------------------------Gender Results (pie chart)----------------------------------------

@app.route("/GenderResults")
@login_required
def GenderResults():
    try:
    # Load CSV data
        df = pd.read_csv(f"user_{current_user.id}_detections.csv")  
    
    except pd.errors.EmptyDataError:
        flash("No current data available for Gender Analysis. Please ensure customer feedback data is collected.", "warning")
        return render_template("GenderResults.html", labels=[], positive_values=[], negative_values=[], neutral_values=[], heatmap_labels=[], heatmap_datasets=[])
    
    except FileNotFoundError:
        flash("No current data available for Gender Analysis. Please ensure customer feedback data is collected.", "warning")
        return render_template("GenderResults.html", labels=[], positive_values=[], negative_values=[], neutral_values=[], heatmap_labels=[], heatmap_datasets=[])

    # Ensure the columns exist in the dataframe
    if "Timestamp" in df.columns and "Emotion" in df.columns and "Gender" in df.columns:
        labels = df["Timestamp"].tolist()
        emotions = df["Emotion"].tolist()
        genders = df["Gender"].tolist()

        # Calculate emotion counts
        male_count = genders.count('Male')
        female_count = genders.count('Female')

        # Calculate total count and percentages
        total_count = male_count + female_count
        if total_count > 0:
            male_percentage = round((male_count / total_count) * 100, 2)
            female_percentage = round((female_count / total_count) * 100, 2)
        else:
            male_percentage = 0
            female_percentage = 0
    else:
        labels = []
        emotions = []
        genders = []
        male_count = 0
        female_count = 0
        male_percentage = 0
        female_percentage = 0

    return render_template("GenderResults.html", labels=labels, emotions=emotions, genders=genders,
                           male_count=male_count, female_count=female_count, 
                           total_count=total_count,
                           male_percentage=male_percentage, female_percentage=female_percentage)

# -----------------------------------------------------MaleEmotionResults---------------------------------------------------

@app.route("/MaleEmotionResults")
@login_required
def MaleEmotionResults():
    try:
    # Load CSV data
        df = pd.read_csv(f"user_{current_user.id}_detections.csv")  

    except pd.errors.EmptyDataError:
        flash("No current data available for Male Emotion Analysis. Please ensure customer feedback data is collected.", "warning")
        return render_template("MaleEmotionResults.html", labels=[], positive_values=[], negative_values=[], neutral_values=[], heatmap_labels=[], heatmap_datasets=[])

    except FileNotFoundError:
        flash("No current data available for Male Emotion Analysis. Please ensure customer feedback data is collected.", "warning")
        return render_template("MaleEmotionResults.html", labels=[], positive_values=[], negative_values=[], neutral_values=[], heatmap_labels=[], heatmap_datasets=[])

    # Ensure the columns exist in the dataframe
    if "Timestamp" in df.columns and "Emotion" in df.columns and "Gender" in df.columns:
        # Filter data for males
        male_df = df[df["Gender"] == "Male"]
        labels = male_df["Timestamp"].tolist()
        emotions = male_df["Emotion"].tolist()
        genders = male_df["Gender"].tolist()

        # Calculate emotion counts
        male_positive_count = emotions.count('positive')
        male_negative_count = emotions.count('negative')
        male_neutral_count = emotions.count('neutral')

        # Calculate total count and percentages
        male_total_count = male_positive_count + male_negative_count + male_neutral_count
        if male_total_count > 0:
            male_positive_percentage = round((male_positive_count / male_total_count) * 100, 2)
            male_negative_percentage = round((male_negative_count / male_total_count) * 100, 2)
            male_neutral_percentage = round((male_neutral_count / male_total_count) * 100, 2)
        else:
            male_positive_percentage = 0
            male_negative_percentage = 0
            male_neutral_percentage = 0

        if male_total_count == 0:
            flash("No male participants found in the data.", "warning") 


    else:
        labels = []
        emotions = []
        genders = []
        male_positive_count = 0
        male_negative_count = 0
        male_neutral_count = 0
        male_total_count = 0
        male_positive_percentage = 0
        male_negative_percentage = 0
        male_neutral_percentage = 0

    return render_template("MaleEmotionResults.html", labels=labels, emotions=emotions, genders=genders,
                           male_positive_count=male_positive_count, male_negative_count=male_negative_count, male_neutral_count=male_neutral_count,
                           male_positive_percentage=male_positive_percentage, male_negative_percentage=male_negative_percentage, male_neutral_percentage=male_neutral_percentage,
                           male_total_count=male_total_count)

# -----------------------------------------------------------FemaleEmotionResults------------------------------------------

@app.route("/FemaleEmotionResults")
@login_required
def FemaleEmotionResults():
    try:
    # Load CSV data
        df = pd.read_csv(f"user_{current_user.id}_detections.csv")  

    except pd.errors.EmptyDataError:
        flash("No current data available for Female Emotion Analysis. Please ensure customer feedback data is collected.", "warning")
        return render_template("FemaleEmotionResults.html", labels=[], positive_values=[], negative_values=[], neutral_values=[], heatmap_labels=[], heatmap_datasets=[])
    
    except FileNotFoundError:
        flash("No current data available for Female Emotion Analysis. Please ensure customer feedback data is collected.", "warning")
        return render_template("FemaleEmotionResults.html", labels=[], positive_values=[], negative_values=[], neutral_values=[], heatmap_labels=[], heatmap_datasets=[])

    # Ensure the columns exist in the dataframe
    if "Timestamp" in df.columns and "Emotion" in df.columns and "Gender" in df.columns:
        # Filter data for males
        female_df = df[df["Gender"] == "Female"]
        labels = female_df["Timestamp"].tolist()
        emotions = female_df["Emotion"].tolist()
        genders = female_df["Gender"].tolist()

        # Calculate emotion counts
        female_positive_count = emotions.count('positive')
        female_negative_count = emotions.count('negative')
        female_neutral_count = emotions.count('neutral')

        # Calculate total count and percentages
        female_total_count = female_positive_count + female_negative_count + female_neutral_count
        if female_total_count > 0:
            female_positive_percentage = round((female_positive_count / female_total_count) * 100, 2)
            female_negative_percentage = round((female_negative_count / female_total_count) * 100, 2)
            female_neutral_percentage = round((female_neutral_count / female_total_count) * 100, 2)
        else:
            female_positive_percentage = 0
            female_negative_percentage = 0
            female_neutral_percentage = 0

        if female_total_count == 0:
            flash("No female participants found in the data.", "warning") 

    else:
        labels = []
        emotions = []
        genders = []
        female_positive_count = 0
        female_negative_count = 0
        female_neutral_count = 0
        female_total_count = 0
        female_positive_percentage = 0
        female_negative_percentage = 0
        female_neutral_percentage = 0

    return render_template("FemaleEmotionResults.html", labels=labels, emotions=emotions, genders=genders,
                           female_positive_count=female_positive_count, female_negative_count=female_negative_count, female_neutral_count=female_neutral_count,
                           female_positive_percentage=female_positive_percentage, female_negative_percentage=female_negative_percentage, female_neutral_percentage=female_neutral_percentage,
                           female_total_count=female_total_count)

# -----------------------------------------------------------AnalyticalAnalysis---------------------------------------------

@app.route("/AnalyticalAnalysis")
@login_required
def AnalyticalAnalysisverTime():

# Save to local CSV (All feedbacks, with current feedback and history)
    try:
        user_detections_ref = db.collection('users').document(current_user.id).collection('detections')
        docs = user_detections_ref.stream()
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            data.append([doc_data['timestamp'], doc_data['emotion'], doc_data['gender']])
        # Save to export CSV 
        export_filename = f"user_{current_user.id}_export.csv"
        with open(export_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Emotion', 'Gender'])
            writer.writerows(data)
        
        print(f"Exported {len(data)} records to {export_filename}")
    except Exception as e:
        print(f"Export failed: {str(e)}")

#Data display in Analytical Analysis
    try:
        df = pd.read_csv(f"user_{current_user.id}_export.csv")  # CSV file
        if df.empty:
            flash("No Historical Data Available for Analytical Analysis.", "warning")
            return render_template("AnalyticalAnalysis.html", labels=[], positive_values=[], negative_values=[],neutral_values=[], heatmap_labels=[], heatmap_datasets=[])

    except FileNotFoundError:
        flash("No data available for Analytical Analysis.", "warning")
        return render_template("AnalyticalAnalysis.html", labels=[], positive_values=[], negative_values=[], neutral_values=[], heatmap_labels=[], heatmap_datasets=[])

    # Ensure the columns exist in the dataframe
    if "Timestamp" in df.columns and "Emotion" in df.columns and "Gender" in df.columns:
        # Convert Timestamp to datetime
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Categorize time of day
        df["TimeOfDay"] = df["Timestamp"].dt.hour.apply(lambda x: "Morning" if 5 <= x < 12 else "Afternoon" if 12 <= x < 17 else "Evening" if 17 <= x < 21 else "Night")

        # Create a new column with Date and TimeOfDay
        df["DateTimeOfDay"] = df["Timestamp"].dt.date.astype(str) + " " + df["TimeOfDay"]

        # Group by DateTimeOfDay and Emotion
        grouped = df.groupby(["DateTimeOfDay", "Emotion"]).size().unstack(fill_value=0)

        # Prepare data for the bar chart
        labels = grouped.index.tolist()
        positive_values = grouped["positive"].tolist() if "positive" in grouped else [0] * len(labels)
        negative_values = grouped["negative"].tolist() if "negative" in grouped else [0] * len(labels)
        neutral_values = grouped["neutral"].tolist() if "neutral" in grouped else [0] * len(labels)

        # Heatmap Data
        df["Date"] = df["Timestamp"].dt.date  # Extract Date
        heatmap_data = df.pivot_table(index="Date", columns="Emotion", values="Gender", aggfunc="count").fillna(0)
        heatmap_labels = heatmap_data.index.tolist()
        heatmap_datasets = [{"label": col, "data": heatmap_data[col].tolist()} for col in heatmap_data.columns]

        # ----------Total Emotion and Gender Pie Chart--------- 

        # Extract data from the dataframe for display pie chart
        emotions = df["Emotion"].tolist()
        genders = df["Gender"].tolist()

        # Calculate All emotion counts
        positive_count = emotions.count('positive')
        negative_count = emotions.count('negative')
        neutral_count = emotions.count('neutral')
        # Calculate total count
        total_count = positive_count + negative_count + neutral_count

        # Calculate All gender counts
        male_count = genders.count('Male')
        female_count = genders.count('Female')

        # ----------Total Emotion Distribution based on Gender Pie Chart---------

        # Filter data for males
        male_df = df[df["Gender"] == "Male"]
        emotions = male_df["Emotion"].tolist()
        genders = male_df["Gender"].tolist()

        # Calculate emotion counts
        male_positive_count = emotions.count('positive')
        male_negative_count = emotions.count('negative')
        male_neutral_count = emotions.count('neutral')
        # Calculate total count and percentages
        male_total_count = male_positive_count + male_negative_count + male_neutral_count

        # Filter data for females
        female_df = df[df["Gender"] == "Female"]
        emotions = female_df["Emotion"].tolist()
        genders = female_df["Gender"].tolist()

        # Calculate emotion counts
        female_positive_count = emotions.count('positive')
        female_negative_count = emotions.count('negative')
        female_neutral_count = emotions.count('neutral')
         # Calculate total count and percentages
        female_total_count = female_positive_count + female_negative_count + female_neutral_count

    else:
        labels = []
        positive_values = []
        negative_values = []
        neutral_values = []
        heatmap_labels = []
        heatmap_datasets = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        male_count = 0
        female_count = 0
        male_positive_count = 0
        male_negative_count = 0
        male_neutral_count = 0
        male_total_count = 0
        female_positive_count = 0
        female_negative_count = 0
        female_neutral_count = 0
        female_total_count = 0

    return render_template("AnalyticalAnalysis.html", labels=labels, positive_values=positive_values, negative_values=negative_values, neutral_values=neutral_values,heatmap_labels=heatmap_labels,
                           heatmap_datasets=heatmap_datasets,
                           total_count=total_count,
                           positive_count=positive_count, negative_count=negative_count, neutral_count=neutral_count,
                           male_count=male_count, female_count=female_count,
                           male_positive_count=male_positive_count, male_negative_count=male_negative_count, male_neutral_count=male_neutral_count,
                           male_total_count=male_total_count,
                           female_positive_count=female_positive_count, female_negative_count=female_negative_count, female_neutral_count=female_neutral_count,
                           female_total_count=female_total_count)


# --------------------------------------------------------------contact us--------------------------------------------------

app.secret_key = 'your_secret_key'

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'ause3080@gmail.com'
app.config['MAIL_PASSWORD'] = 'ado4neka'

mail = Mail(app)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/send_email', methods=['POST'])
def send_email():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']

    # Compose the email
    msg = Message(
        subject=f"Contact Form Message from {name}",
        sender=app.config['MAIL_USERNAME'],
        recipients=['ause3080@gmail.com'],  # facecx email
        body=f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
    )

    try:
        mail.send(msg)
        flash('Message sent successfully!', 'success')
    except Exception as e:
        print(str(e))
        flash('Failed to send message. Please try again later.', 'danger')

    # return redirect('/contact')
    return render_template('contact.html')

# -------------------------------------------------bg-animation---------------------------------------------------

@app.route("/animation")
def animation():
    return render_template("animation.html")


if __name__ == '__main__':
    app.run(debug=True)