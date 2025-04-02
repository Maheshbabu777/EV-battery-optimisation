from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import confusion_matrix,accuracy_score

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
df = pd.read_csv("dataset/ev_battery_charging_data.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/graph_page")
def graph_page():
    return render_template("graph.html")

@app.route("/confusion_matrix_page")
def confusion_matrix_page():
    return render_template("confusion_matrix.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_input = request.get_json()
        if "features" not in data_input:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = data_input["features"]

        if not isinstance(features, list) or len(features) != 12:
            return jsonify({"error": "Invalid input! Expected 12 numeric values."}), 400

        input_data = np.array(features, dtype=np.float64).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)

        class_messages = {
            0: "Charging efficiency is low. Consider optimizing the charging cycle.",
            1: "Charging is moderate. Battery performance is stable.",
            2: "Charging is optimal. No issues detected."
        }

        message = class_messages.get(int(prediction[0]), "Unknown class. Check input values.")

        return jsonify({"Optimal Charging Duration Class": int(prediction[0]), "Message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/graph")
def get_graph():
    try:
        if "Charging Cycles" not in df.columns or "Degradation Rate (%)" not in df.columns:
            return jsonify({"error": "Required columns missing in dataset!"}), 400

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=df["Charging Cycles"], y=df["Degradation Rate (%)"], color="blue", alpha=0.6)
        plt.xlabel("Charging Cycles")
        plt.ylabel("Degradation Rate (%)")
        plt.title("Battery Degradation Scatter Plot")
        plt.grid(True)

        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode()

        return jsonify({"graph": img_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/confusion_matrix")
def get_confusion_matrix():
    try:
        y_test = np.load("model/y_test.npy")
        y_pred = np.load("model/y_pred.npy")

        
        with open("model/metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
        accuracy = metrics.get("accuracy", None)

        if accuracy is None:
            return jsonify({"error": "Accuracy not found in metrics.pkl"}), 400

        return jsonify({"accuracy": round(accuracy * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
