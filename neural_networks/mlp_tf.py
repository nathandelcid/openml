from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

class MLP():
    def __init__(self):
        self.model = Sequential( [
            InputLayer(shape=(2,)),
            Dense(50, activation = "relu"),
            Dense(50, activation = "relu"),
            Dense(1, activation="sigmoid")
        ] )
    
    def compile(self, optimizer="adam", loss="binary_crossentropy"):
        self.model.compile(optimizer=optimizer, loss=loss)
        
    def fit(self, X, y, epochs=20):
        self.model.fit(X, y, epochs=epochs)

    def evaluate(self, X, y, verbose=0):
        return self.model.evaluate(X, y, verbose=verbose)