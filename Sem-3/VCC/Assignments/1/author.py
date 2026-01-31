from flask import Flask
import os

app = Flask(__name__)

AUTHOR_NAME = "Gaurav Singh"

@app.route("/author")
def author():
    return f"By: {AUTHOR_NAME}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
