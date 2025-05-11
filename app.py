from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load Model and Scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/products")
def product():
    return render_template("product.html")

@app.route("/consult_doctor")
def consult():
    return render_template("doc.html")

@app.route("/sign_in")
def sign_in():
    return render_template("signin.html")

@app.route("/sign_up")
def sign_up():
    return render_template("signup.html")

@app.route("/book_your_appointment")
def book():
    return render_template("byt.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            Pregnancies = float(request.form["Pregnancies"])
            Glucose = float(request.form["Glucose"])
            BMI = float(request.form["BMI"])
            Age = float(request.form["Age"])
            DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])

            data = [[Pregnancies, Glucose, BMI, Age, DiabetesPedigreeFunction]]
            data_scaled = scaler.transform(data)

            prediction = model.predict(data_scaled)
            probability = model.predict_proba(data_scaled)[0][1] * 100  # Risk in percent
            risk_percent = round(probability, 2)

            if prediction[0] == 1:
                result = "DIABETIC 🚨"
                color = "red-box"
                suggestion = [
                    "1.) Start Low Carb Diet",
                    "2.) Walk Daily for 30 Minutes 🚶‍♂️",
                    "3.) Avoid Sugar & Junk Foods",
                    "4.) Drink 3-4 Litres of Water 💧",
                    "5.) Regular Checkups (HbA1c Test) 📅",
                    "6.) Meditation for Stress Control 🧠"
                ]
            else:
                result = "NON-DIABETIC ✅"
                color = "green-box"
                suggestion = [
                    "1.) Maintain Balanced Diet 🍎🥦",
                    "2.) Exercise Regularly 🏃‍♀️",
                    "3.) Drink Enough Water 💦",
                    "4.) Avoid Processed Sugars",
                    "5.) Do Regular Blood Checkup",
                    "6.) Stay Happy, Stay Healthy 🔥🚀"
                ]

            return render_template("result.html", result=result, color=color, suggestion=suggestion, risk=risk_percent)

        except (ValueError, KeyError):
            return render_template("index.html", error="Invalid input. Please enter valid numbers.")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
