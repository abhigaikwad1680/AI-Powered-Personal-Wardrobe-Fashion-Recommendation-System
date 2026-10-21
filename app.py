from flask import Flask, jsonify, render_template, request
from recommender import recommend

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/recommend")
def api_recommend():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()[:300]
    gender = data.get("gender", "men")
    weather = data.get("weather", "any")
    if not query:
        return jsonify(error="Please describe the occasion and the look you want."), 400
    if gender not in ("men", "women") or weather not in ("any", "summer", "winter", "rainy", "all-season"):
        return jsonify(error="Invalid options."), 400
    return jsonify(results=recommend(query, gender, weather))

if __name__ == "__main__":
    app.run(debug=True)
