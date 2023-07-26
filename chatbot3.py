from keras.models import load_model
from keras.layers import Dense
from keras.models import Sequential
import warnings
import pickle
import json
import numpy as np
import tensorflow as tf
import random
import os

# Usde to for Contextualisation and Other NLP Tasks.
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
stemmer = LancasterStemmer()

# Other
warnings.filterwarnings("ignore")

print("Processing the Intents.....")
try:
    with open('intents.json') as json_data:
        intents = json.load(json_data)
except FileNotFoundError:
    print("Error: intents.json file not found. Please make sure the file exists.")
    exit(1)

documents = []
classes = []
ignore_words = ['?']

print("Looping through the Intents to Convert them to words, classes, documents and ignore_words.......")
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        print(f"tokenized words: {w}")
        # add to documents in our corpus
        documents.append((' '.join(w), intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print("Creating the Data for our Model.....")
training = []
output = []

print("Creating Traning Set and Tf-idf Representation....")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Create Tf-idf representation for all the sentences
tfidf_matrix = tfidf_vectorizer.fit_transform(
    [doc[0] for doc in documents]).toarray()

# Get the fixed size for all Tf-idf vectors
tfidf_size = tfidf_matrix.shape[1]

for i, doc in enumerate(documents):
    bag = tfidf_matrix[i].tolist()
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

print("Shuffling Randomly and Converting into Numpy Array for Faster Processing......")
random.shuffle(training)

# Separate the bag of words and output rows
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

# Pad the sequences to the maximum length
train_x = pad_sequences(train_x, maxlen=tfidf_size, dtype='float32')

print("Building Neural Network for Out Chatbot to be Contextual....")
print("Resetting graph data....")
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(len(train_x[0]),)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])


print("Compiling the Model.......")
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model_path = 'model.h5'

if not os.path.exists(model_path):
    print("Training the Model.......")
    model.fit(np.array(train_x), np.array(train_y),
              epochs=1000, batch_size=8, verbose=1)
    print("Saving the Model.......")
    model.save(model_path)
else:
    print("Loading the pre-trained Model......")
    # Load our saved model
    model = tf.keras.models.load_model(model_path)


print("Pickle is also Saved..........")
try:
    pickle.dump({'tfidf_vectorizer': tfidf_vectorizer, 'classes': classes, 'train_x': train_x, 'train_y': train_y},
                open("training_data", "wb"))
except Exception as e:
    print("Error occured while saving the pickle data: ", e)

print("Loading Pickle.....")
try:
    data = pickle.load(open("training_data", "rb"))
    tfidf_vectorizer = data['tfidf_vectorizer']
    classes = data['classes']
    train_x = data['train_x']
    train_y = data['train_y']
except Exception as e:
    print("Error occured while loading pickle data: ", e)
    exit(1)

# load our saved model
print("Loading the Model......")
try:
    model = model = tf.keras.models.load_model('model.h5')
except Exception as e:
    print("Error occured while loading the model: ", e)
    exit(1)


def clean_up_sentence(sentence):
    # It Tokenizes or Breaks it into the constituent parts of the Sentence.
    sentence_words = nltk.word_tokenize(sentence)
    # Stemming means to find the root of the word.
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return ' '.join(sentence_words)


ERROR_THRESHOLD = 0.25
print("ERROR_THRESHOLD = 0.25")


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def classify(sentence):
    # Preprocess the user's sentence and convert it to Tf-idf representation
    sentence_tfidf = tfidf_vectorizer.transform(
        [clean_up_sentence(sentence)]).toarray()
    # Prediction to Get the Probability from the Model
    results = model.predict([sentence_tfidf])[0]
    # Exclude those results which are Below Threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # Sorting is Done because higher Confidence Answer comes first.
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # That Means if Classification is Done then Find the Matching Tag.
    if results:
        # Long Loop to get the Result.
        while results:
            for i in intents['intents']:
                # Tag Finding
                if i['tag'] == results[0][0]:
                    # Random Response from High Order Probabilities
                    return print(random.choice(i['responses']))

            results.pop(0)


while True:
    input_data = input("You- ")
    answer = response(input_data)
    answer
