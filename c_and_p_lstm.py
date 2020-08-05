# I'll get back to the Irish Times after this, but this will work to help me
# get through the tutorial and get a better understanding of what I'm up against
# here.
import sys
import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def tokenize_words(input):
    # Lower the case of stuff
    input = input.lower()

    # Set up the tokenizer (1 or more word characters)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # Filter out the stop words
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return ' '.join(filtered)


# Read the file and tokenize the words (It reads much faster than I read it.)
with open('data/crime_and_punishment.txt') as f:
    file = f.read()

processed_txt = tokenize_words(file)

# Sort the characters and convert to numbers
chars = sorted(list(set(processed_txt)))
char_to_num = dict((c, i) for i, c in enumerate(chars))

# Get length of input and of vocabulary
input_len = len(processed_txt)
vocab_len = len(chars)
print('Total number of characters: ', input_len)
print('Total vocab: ', vocab_len)

seq_length = 100
x_data, y_data = [], []

for i in range(0, input_len - seq_length, 1):
    # Define the moving window
    # For each character, we get the next n in seq_length
    in_seq = processed_txt[i:i + seq_length]

    # The next character after the sequence (the predicted value)
    out_seq = processed_txt[i + seq_length]

    # Convert characters from sequences into numbers for learning purposes
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print('Total patterns: ', n_patterns)

# Not sure why we're reshaping this way
X = np.reshape(x_data, (n_patterns, seq_length, 1))

# Converting everything to floats between 0 and 1 so sigmoid function works
X = X / float(vocab_len)

# Convert y_data to categorical. It would appear that we're effectively
# making this a classification task with 50 classes. I wonder what would
# happen if we just made these floats as well...could we do something like
# a regression? The tutorial says we're one-hot encoding here...and we are!
y = np_utils.to_categorical(y_data)

# MODEL TIME!
model = Sequential()

# Why this number of units? What is return_sequences?
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))

# Why these dropout values? Arbitrary?
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam')

# Save weights and reload when the training is done?
# What is it to use these as callbacks? So much to learn yet...
path = 'model_weights_saved.hd5'
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1,
                             save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

Fit the model!
model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)

# All of this comes after a previous run, reloading the weights file from before.
fname = 'model_weights_saved.hd5'
model.load_weights(fname)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Well dang, I guess we need to convert from numbers back to characters now,
# if we want any results that mean anything.
num_to_char = dict((i, c) for i, c in enumerate(chars))

# Picking a random number that is then used to select a character from the x_data
# to use as a seed for pattern generation.
# Needs to be the width of x_data so that we have something to look back to.
start = np.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print('Random seed: \"{}\"'.format(''.join([num_to_char[val] for val in pattern])))

# Make predictions for the next 1000 characters
for _ in range(1000):
    # Reshape pattern into something that can be used for predictions
    x = np.reshape(pattern, (1, len(pattern), 1))

    # Convert pattern values to values between 0 and 1 (probabilties?)
    # Shouldn't we already have a pick from the softmax activation above?
    x = x / float(vocab_len)

    # Generate a prediction from the pattern
    prediction = model.predict(x, verbose=0)

    # Get the index of the highest value
    index = np.argmax(prediction)

    # Convert the index to the predicted character
    result = num_to_char[index]

    sys.stdout.write(result)

    # Add the newly-predicted character to the end of the pattern and shift the
    # window one character forward for the next prediction.
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
