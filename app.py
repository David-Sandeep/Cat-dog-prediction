import os
from flask import Flask, render_template, request, send_from_directory
from keras.preprocessing import image
from keras.preprocessing.image import load_img

from keras.models import load_model
import numpy as np
import tensorflow as tf

app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'


def load__model():
    """Load model once at running time for all the predictions"""
    print('[INFO] : Model loading ................')
    global model
    model = load_model(MODEL_FOLDER + '/cat_dog_classifierr.h5')
    print('[INFO] : Model loaded')


def predict(fullpath):
    data = load_img(fullpath, target_size=(128, 128, 3))
    test_image = image.img_to_array(data)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    return result


# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else :
        try:
            file = request.files['image']
            fullname = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(fullname)
            
        
            result = predict(fullname)
            ans = np.argmax(result, axis=1)
            if ans==1:
                prediction="dog"
            else:
                prediction = "cat"
        
            return render_template('predict.html', image_file_name=file.filename, label=prediction)
        except Exception as e:
            print(str(e))
            return render_template('index.html')
        
        


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def create_app():
    load__model()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
