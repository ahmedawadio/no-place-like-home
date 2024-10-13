from flask import Flask, jsonify
try: from .database import get_location
except: from database import get_location

try: from .bucket import get_image
except: from bucket import get_image

from api.bucket import get_image

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



@app.route("/api/image/<string:mid>", methods=["GET"])
def image(mid):
    return get_image(mid)


