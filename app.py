from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model, vectorizer = None, None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("Loaded trained model.")
else:
    print("No trained model found. Using fallback keywords.")
    def dummy_predict(texts):
        labels = []
        for t in texts:
            if "win" in t.lower() or "free" in t.lower() or "prize" in t.lower():
                labels.append("Spam")
            else:
                labels.append("Not Spam")
        return labels
    model = None
    vectorizer = None
    predict_func = dummy_predict

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        email_text = request.form.get("email_text", "")
        if model and vectorizer:
            x = vectorizer.transform([email_text])
            pred = model.predict(x)[0]
            prediction = "Spam" if pred == 1 else "Not Spam"
        else:
            prediction = predict_func([email_text])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
