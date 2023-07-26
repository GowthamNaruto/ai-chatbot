from keras.models import load_model
from keras.layers import Dense
from keras.models import Sequential
import warnings
import pickle
import json
import numpy as np
import tensorflow as tf
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
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

words = []
classes = []
documents = []
ignore_words = ['?']
print("Looping through the Intents to Convert them to words, classes, documents and ignore_words.......")
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # print(f"tokenized words: {w}")
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print("Stemming, Lowering and Removing Duplicates.......")
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)

print("Creating the Data for our Model.....")
training = []
output = []
print("Creating an List (Empty) for Output.....")
output_empty = [0] * len(classes)
print("Creating Traning Set, Bag of Words for our Model....")
max_bag_length = 0  # Initialize maximum bag length
training_data = []  # Separate list to store bag and output_row
# List of zeros with length equal to the number of classes
output_empty = [0] * len(classes)
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Update max_bag_length if the current bag is longer
    max_bag_length = max(max_bag_length, len(bag))

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training_data.append((bag, output_row))

# Separate the bag and output_row lists before converting to NumPy array
bags = [data[0] for data in training_data]
output_rows = [data[1] for data in training_data]

print("Shuffling Randomly and Converting into Numpy Array for Faster Processing......")
# Convert bags and output_rows lists to NumPy arrays
bags_np = np.array(bags)
output_rows_np = np.array(output_rows)

# Create a single 2D NumPy array by combining bags_np and output_rows_np along the second axis (columns)
training = np.column_stack((bags_np, output_rows_np))

print("Creating Train and Test Lists.....")
train_x = training[:, :-len(output_empty)]
train_y = training[:, -len(output_empty):]
print("Building Neural Network for Our Chatbot to be Contextual....")
print("Resetting graph data....")
tf.compat.v1.reset_default_graph()


model = Sequential()
model.add(Dense(8, input_dim=max_bag_length, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

try:
    print("Training....")
    model.fit(train_x, train_y, epochs=1000, batch_size=8, verbose=1)
except Exception as e:
    print("Error occured during model training: ", e)
    exit(1)

print("Saving the Model.......")
try:
    model.save('model.keras')
except Exception as e:
    print("Error occured while saving the model: ", e)
    exit(1)


print("Pickle is also Saved..........")
try:
    pickle.dump({'words': words, 'classes': classes, 'train_x': train_x,
                'train_y': train_y}, open("training_data", "wb"))
except Exception as e:
    print("Error occured while saving the pickle data: ", e)

print("Loading Pickle.....")
try:
    data = pickle.load(open("training_data", "rb"))
    words = data['words']
    classes = data['classes']
    train_x = data['train_x']
    train_y = data['train_y']
except Exception as e:
    print("Error occured while loading pickle data: ", e)
    exit(1)

# load our saved model
print("Loading the Model......")
try:
    model = load_model('./model.keras')
except Exception as e:
    print("Error occured while loading the model: ", e)
    exit(1)


def clean_up_sentence(sentence):
    # It Tokenizes or Breaks it into the constituent parts of the Sentence.
    sentence_words = nltk.word_tokenize(sentence)
    # Stemming means to find the root of the word.
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


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
    # Prediction or To Get the Possibility or Probability from the Model
    results = model.predict(np.array([bow(sentence, words)]))[0]
    # Exclude those results which are Below Threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # Sorting is Done because higher Confidence Answer comes first.
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        # Tuple -> Intent and Probability
        return_list.append((classes[r[0]], r[1]))
    return return_list


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
