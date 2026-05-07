from flask import Flask, render_template, request, redirect, url_for, session
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import sqlite3
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fruit_disease_secret_key")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DB_NAME = "users.db"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

model = load_model("keras_model.h5", compile=False)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch"
]

np.set_printoptions(suppress=True)

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized
    return data

def fuzzy_severity(confidence):
    return fuzzy_severity_with_disease(confidence, None)

def fuzzy_severity_with_disease(confidence, disease):
    DISEASE_SEVERITY_MOD = {
        "Blueberry___healthy": -30,
        "Grape___healthy": -30,
        "Strawberry___healthy": -30,
        "Apple___Apple_scab": 0,
        "Cherry_(including_sour)___Powdery_mildew": 0,
        "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": 5,
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 5,
        "Strawberry___Leaf_scorch": 0,
        "Grape___Black_rot": 20,
        "Grape___Esca_(Black_Measles)": 20,
    }
    base_mod = DISEASE_SEVERITY_MOD.get(disease, 0)
    adjusted = max(0.0, min(100.0, confidence + base_mod))
    if adjusted < 50:
        return "Low / குறைந்த"
    elif adjusted < 80:
        return "Medium / நடுத்தர"
    else:
        return "High / அதிகம்"

def get_remedy(disease):
    prompt = f"""
    Explain the plant disease '{disease}'.
    Give remedies and prevention steps.
    Provide output in English and Tamil.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        try:
            conn = get_db()
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Username already exists")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password)).fetchone()
        conn.close()
        if user:
            session["user"] = username
            return redirect(url_for("predict"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    result = confidence = severity = remedy = image_path = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)
            data = preprocess_image(image_path)
            prediction = model.predict(data)
            idx = np.argmax(prediction)
            result = CLASS_NAMES[idx]
            confidence = float(prediction[0][idx]) * 100
            try:
                severity = fuzzy_severity_with_disease(confidence, result)
            except Exception:
                severity = fuzzy_severity(confidence)
            remedy = get_remedy(result)
    return render_template("index.html", prediction=result, confidence=confidence,
                           severity=severity, remedy=remedy, image_path=image_path)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=False)
