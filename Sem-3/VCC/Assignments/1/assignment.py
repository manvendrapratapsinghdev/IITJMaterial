from flask import Flask
import urllib.request
import urllib.error

AUTHOR_API_BASE = "http://192.168.1.48:5002"

AUTHOR_ENDPOINT = f"{AUTHOR_API_BASE}/author"

app = Flask(__name__)

ASSIGNMENT_NAME = "VCCS Assignment 1"

@app.route("/assignment")
def assignment():
    assignment_text = f"Assignment: {ASSIGNMENT_NAME}"
    author_text = _fetch_author_text()
    return f"{assignment_text} {author_text}"

def _fetch_author_text() -> str:
    try:
        with urllib.request.urlopen(AUTHOR_ENDPOINT, timeout=2) as resp:
            return resp.read().decode("utf-8").strip() or "By: Unknown"
    except urllib.error.URLError:
        return "By: Unknown"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
