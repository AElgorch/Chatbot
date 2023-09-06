import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

# Load the intents data from a JSON file
with open("intents.json") as file:
    data = json.load(file)

try:
    # Try to load preprocessed data from a pickle file
    with open("data.pickle", "rb") as f:
        text, labels, training, output = pickle.load(f)
except:
    # If the pickle file doesn't exist, preprocess the data
    text = []
    labels = []
    docs_x = []
    docs_y = []

    # Iterate through intents and patterns in the JSON data
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            text.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Tokenize and stem words, remove duplicates, and sort
    text = [stemmer.stem(w.lower()) for w in text if w != "?"]
    text = sorted(list(set(text)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # Create training and output data
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in text:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Convert training and output data to NumPy arrays
    training = numpy.array(training)
    output = numpy.array(output)

    # Save preprocessed data to a pickle file
    with open("data.pickle", "wb") as f:
        pickle.dump((text, labels, training, output), f)

# Create the neural network model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    # Try to load a pre-trained model
    model.load("model.tflearn")
except:
    # Train the model if a pre-trained model is not found
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Define a function to convert user input to a bag of words
def bag_of_words(s, text):
    bag = [0 for _ in range(len(text))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(text):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

# Create a chat function to interact with the bot
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, text)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

# Start the chat function
chat()
