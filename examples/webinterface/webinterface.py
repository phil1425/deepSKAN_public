# deepSKAN
# This is and example implementation using a Flask server as a web interface
# A csv file can be uploaded and will be analyzed by the network.
# The graph of the predicted model gets sent back as an image

from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from deepGTA import DLAnalyzer, GTAnalyzer
import keras
import keras.backend as K
import os
from deepGTA.utils import top_10_acc

# use CPU since we only predict one dataset at a time and probably need the 
# GPU for something else
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)

# Load the model once at the beginning since it takes a few seconds 
# cannot use the DLAnalyzer for this, since keras doesnt like getting 
# loaded from a package as a global variable for flask (???)
@app.before_first_request
def initialize():
   global model
   keras.metrics.top_10_acc = top_10_acc

   model = keras.models.load_model('RES-33L-15M-103C-5S-6-48E-53A')
   model.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adam(lr=0.002, beta_1=0.9, 
               beta_2=0.999, epsilon=0.01, amsgrad=False),
               metrics=['accuracy', top_10_acc])
   model._make_predict_function()

# Send Upload form
@app.route('/')
def upload_file():
    return render_template('index.html')

# Upload csv to the sever, generate prediction and send predicted model back
# as png
@app.route('/result', methods = ['POST'])
def uploader_file():
    dl_model = DLAnalyzer(load_model=False)
    dl_model.model = model
    if request.method == 'POST':
        f = request.files['file']
        name = f.filename
        f.save('examples/webinterface/file1')

        X = np.loadtxt('examples/webinterface/file1')

        plt.figure()
        plt.imshow(X)
        plt.savefig('examples/webinterface/static/img_1.png')
        plt.close()

        X = np.reshape(X, [1, 256, 64, 1])

        P = dl_model.predict(X)

        n = dl_model.draw_prediction(P, 
                            path='examples/webinterface/static/prob.png', 
                            min_confidence=0.7)
    return render_template('result.html', name=name, n=n)

if __name__ == "__main__":
    app.run(debug=True)