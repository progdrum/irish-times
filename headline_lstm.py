import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint

# Make this bitch reproducible
np.random.seed(777)

df = pd.read_csv("data/irishtimes-date-text.csv")
longest_headline = int(df['headline_text'].str.len().max())
cats_and_heads = df.drop(['publish_date'], axis=1)
cols = cats_and_heads.columns.tolist()
cols = cols[-1:] + cols[:-1]
cats_and_heads = cats_and_heads[cols]
cats_and_heads.dropna(how='any', inplace=True)
cats_and_heads['headline_category'] = cats_and_heads['headline_category'].apply(
    lambda hline: hline.split('.')[0])
X_train, X_test, y_train, y_test = train_test_split(cats_and_heads['headline_text'],
                                                    cats_and_heads['headline_category'],
                                                    test_size=0.3)

# Get vocabulary size
vocab_size = len(set(' '.join(cats_and_heads['headline_text'].to_list()).split(' ')))

# Tokenize the headlines to make them into nifty numbers
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(cats_and_heads['headline_text'])

# Pad the sequences so that they're all the same length
# Apparently I need to be handing the text to change as a list of texts

# Why does the X_test size appear to change?
X_tr = tokenizer.texts_to_sequences(X_train.to_list())
X_tst = tokenizer.texts_to_sequences(X_test)

# "Pre" pads by default. Consider trying "post" padding.
X_tr = sequence.pad_sequences(X_tr, maxlen=longest_headline)
X_tst = sequence.pad_sequences(X_tst, maxlen=longest_headline)

# One-hot encode the category labels
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# Let's save the checkpoints
checkpoint = ModelCheckpoint('headline_checkpoints.md5', monitor='loss',
                             verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

# Model time!
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=longest_headline))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Run it!
model.fit(X_tr, y_train, epochs=3, batch_size=32, callbacks=desired_callbacks)

# Reload the saved model and classify the headlines
model.load_weights('headline_checkpoints.md5')
model.compile(loss='categorical_crossentropy', optimizer='adam')

scores = model.evaluate(X_tst, y_test)
print(f'Accuracy: {scores[1] * 100}')

predictions = model.predict_classes(X_tst)
