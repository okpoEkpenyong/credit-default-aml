from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

MODEL_URI = "models:/local_model/1"  # or the latest model version
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.route("/")
def home():
    return {"status": "API running", "version": "v1"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        preds = model.predict(df)
        return jsonify({"prediction": int(preds[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
