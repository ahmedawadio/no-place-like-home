from flask import Flask, jsonify
try: from .database import get_location
except: from database import get_location

app = Flask(__name__)

@app.route("/api/python")
def hello_world():
    return "Hello, World!"




@app.route("/api/zipcode/<string:zipcode>", methods=["GET"])
def location(zipcode):
    if len(zipcode) != 5 or not zipcode.isdigit():
        return jsonify({"error": "Must be a 5-digit zip code."}), 400

    response = jsonify(get_location(zipcode))
    return response

