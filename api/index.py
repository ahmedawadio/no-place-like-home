from flask import Flask, jsonify
# from database import get_location

app = Flask(__name__)

@app.route("/api/python")
def hello_world():
    return "Hello, World!"



# @app.route("/api/location")
# def location():
#     location_data = get_location()
#     return jsonify(location_data)

