from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

models = {
    "model_from_scratch" : tf.keras.models.load_model("static/model_from_scratch.h5"),
    #"resnet_model" : tf.keras.models.load_model("static/resnet_model.h5"),
    #"vgg16_model" : tf.keras.models.load_model("static/vgg16_model.h5")
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def runNeuralNetwork(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    x_test = np.array([img])
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0

    pred = models[model].predict(x_test)[0][0]
    return str(pred * 100)

@app.route('/', methods=['GET'])
def test():
    return 'API working'

@app.route('/infer', methods=['POST'])
def classifyImage():
    if request.method == 'POST':
        resText = ''
        model = request.form.get('model')
        print(model)
        if 'image' not in request.files or not model:
            resText = "Error uploading image"
        else:
            file = request.files['image']
            
            if file and allowed_file(file.filename) and file.filename != '':
                # Read image from memory
                nparr = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                resText = runNeuralNetwork(img, model)
            else:
                resText = "Error uploading image"
        
        response = jsonify({'resText' : resText})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)