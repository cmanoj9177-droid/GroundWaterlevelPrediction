from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = joblib.load("groundwater_model.pkl")
columns = joblib.load("feature_columns.pkl")

def preprocess_input(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    # Ensure all expected columns are present
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]
    return df

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ensure numeric values are cast properly
        for k in data:
            try:    
                data[k] = float(data[k])
            except:
                pass

        processed = preprocess_input(data)
        prediction = model.predict(processed)[0]
        prediction = float(prediction)  # 👈 Convert from float32 to plain float

        if prediction <= 150:
            level = "Safe"
        elif prediction <= 450:
            level = "Moderate"
        else:
            level = "Low"

        return jsonify({
            "prediction": round(prediction, 2),
            "status": level
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
