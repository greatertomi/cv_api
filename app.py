import cloudinary
import cloudinary.uploader
import cloudinary.api
import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from calendar import timegm
from time import gmtime

load_dotenv()

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
cloudinary.config(cloud_name=os.getenv('CLOUD_NAME'), api_key=os.getenv('API_KEY'),
                  api_secret=os.getenv('API_SECRET'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "This is a computer vision API developed by John Oshalusi"})


@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'picture' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file_to_upload = request.files['picture']
    timestamp = timegm(gmtime())

    if file_to_upload and allowed_file(file_to_upload.filename):
        upload_result = cloudinary.uploader.upload(file_to_upload, public_id=f'uel_cv/picture{timestamp}')
        url = upload_result['secure_url']
        # prediction = load_and_predict(url)
        return jsonify({"message": "uploaded successfully", "url": url, "prediction": None})
    else:
        return jsonify({"error": "File has invalid type or does not exist"}), 400


if __name__ == '__main__':
    app.run()
