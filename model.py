# model.py
# Defines the get_model() function which is used for generating the
#   compiled and fitted Keras model object that is ready for testing
# Authors: Jonathan Wilkins and Simon Schoelkopf

# ------------------------------------------------------------------
# Analysis:
#   The main way we arrived at our solution was by adding and
#   removing random things to get an idea of how each part affected
#   the resulting accuracy. After many tests, it seems like more
#   epochs lead to higher accuracy (fairly straight forward
#   considering that you're throwing the same dataset at it hundreds
#   of times). Another thing that improved the accuracy a decent
#   amount was adding the min max scaling to allow the smaller
#   values to be effected more by weighting. On the other hand, we
#   assumed adding more hidden layers to the model would increase
#   the accuracy, but it actually had the opposite effect
#   (drastically). Last but not least, appropriately making the
#   model support multiple classes (one of the first changes we made
#   from the example) took our accuracy from ~17% to around 70%
#   which was a major improvement.
# ------------------------------------------------------------------


# Imports
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# get_model()
#   Returns a Keras model object that is ready to be tested
def get_model():
    # load the dataset
    dataset = loadtxt('breasttissue_train.csv', delimiter=',')

    # split input (x, columns 1-9) and output (y, column 0)
    x = dataset[:,1:]
    y = dataset[:,0]

    # create a MinMaxScaler to scale down the data values
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    # encode class values as ints
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    # convert encoded class values to dummy variables
    dummy_y = np_utils.to_categorical(encoded_y)

    # define Keras model
    model = Sequential()

    # add layers to the Keras model
    model.add(Dense(81, input_dim=9, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # compile the Keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the Keras model on the dataset
    model.fit(x, dummy_y, epochs=300, batch_size=5, verbose=0)

    # evaluate the dataset with the Keras model
    _, accuracy = model.evaluate(x, dummy_y)
    # print('Accuracy: %.2f' % (accuracy*100))

    # return the Keras model
    return model

get_model()
