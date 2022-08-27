import tempfile
from io import BytesIO
from urllib import request
from PIL import Image
import os

import numpy as np
import s3fs
from keras.models import load_model
from keras_preprocessing.image import img_to_array

translate = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')


def get_s3fs():
    return s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)


def s3_get_keras_model(model_name: str):
    with tempfile.TemporaryDirectory() as tempdir:
        s3fs = get_s3fs()
        s3fs.get(f"{BUCKET_NAME}/{model_name}", f"{tempdir}/{model_name}")
        return load_model(f"{tempdir}/{model_name}")


def load_and_predict(url):
    model = s3_get_keras_model('animals_vgg16v260822.hdf5')
    res = request.urlopen(url).read()
    img = Image.open(BytesIO(res)).resize((224, 224))
    test_img = img_to_array(img)
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    return translate[np.argmax(result[0])]
