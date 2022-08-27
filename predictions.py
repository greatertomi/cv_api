from io import BytesIO
from urllib import request
from PIL import Image

import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array

translate = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

model = load_model('../trained_models/animals_vgc16p260822.hdf5')


def load_and_predict(url):
    res = request.urlopen(url).read()
    img = Image.open(BytesIO(res)).resize((224, 224))
    test_img = img_to_array(img)
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    return translate[np.argmax(result[0])]
