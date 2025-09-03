from flask import Flask, request, jsonify, render_template
import os
import joblib

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load or warm start
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"[startup] Loaded model from {MODEL_PATH}")
else:
    print(f"[startup] No model found at {MODEL_PATH}. Start by running: python train.py")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return jsonify({"ok": False, "error": "Model not loaded. Run `python train.py` first."}), 400

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"ok": False, "error": "Missing 'text' field in JSON body."}), 400

    pred = model.predict([text])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        # Return probability for the predicted class
        probs = model.predict_proba([text])[0]
        classes = list(model.classes_)
        proba = float(probs[classes.index(pred)])
    return jsonify({"ok": True, "prediction": pred, "confidence": proba})

if __name__ == "__main__":
    app.run(debug=True)