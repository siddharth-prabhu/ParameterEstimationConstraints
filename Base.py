from abc import ABC, abstractmethod

class Base(ABC):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def score(self):
        pass
    