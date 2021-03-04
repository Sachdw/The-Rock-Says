This is a personal project where I created a simple LSTM RNN model capable of generating short texts in the style of former professional wrestler, the Rock. I created the model by adapting the code found in this tutorial from Tensorflow: 

https://www.tensorflow.org/tutorials/text/text_generation 

The project involved transcribing videos of his old promos as text to create an original text file.
I also used python flask to create a simple web application for the model. You can try it out by visiting the link below and entering any word/phrase. The model will use the word/phrase you entered to try to predict the next character in sequence based on the text file it was trained on.

The project is currently deployed to Heroku as a web app: https://the-rock-says.herokuapp.com/
