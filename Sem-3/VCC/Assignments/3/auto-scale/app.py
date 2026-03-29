from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Auto Scaled Cloud Instance: Assignment by Manvendra Pratap Singh (M25AI2122)"

app.run(host="0.0.0.0", port=5000)