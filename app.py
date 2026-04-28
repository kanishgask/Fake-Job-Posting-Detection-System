from flask import Flask, render_template, request
import pickle
from utils import preprocess_text

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    job_desc = request.form["job_desc"]
    
    processed = preprocess_text(job_desc)
    vector = vectorizer.transform([processed])
    
    prediction = model.predict(vector)[0]
    
    result = "Fake Job ❌" if prediction == 1 else "Real Job ✅"
    
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
