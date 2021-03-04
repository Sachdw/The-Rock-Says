from flask import Flask, render_template, request
import keras
from rock_project import *
from tensorflow import TensorShape


app = Flask(__name__)
# Length of the vocabulary in chars
vocab_size = 72
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024

#Define the model architecture and load pretrained weights
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights('theRock_weights70.h5')
model.build(TensorShape([1, None]))


#Display the main page of the app
@app.route('/')
def home():
    return render_template('index.html')

#When users click the 'generate' button, use the model to generate text
@app.route('/', methods=['POST', 'GET'])
def get_data():
    start = request.form['first_word']
    text = generate_text(model, start)
    return render_template('generated.html',text=text)


if __name__ == "__main__":
    app.run(debug=True)
