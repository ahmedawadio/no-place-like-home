from flask import Flask, jsonify
# from .database import get_location


try: from .database import get_location
except: from database import get_location

app = Flask(__name__)

@app.route("/api/python")
def hello_world():
    return "Hello, World!"


@app.route("/api/location")
def location(zipcode):
    location_data = get_location()
    return jsonify(location_data)

# @app.route("/api/location/<string:zipcode>", methods=["GET"])

