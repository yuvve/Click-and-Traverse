from abc import ABC, abstractmethod


class HumanoidAsyncIO(ABC):
    @abstractmethod
    def send(self, channel, data):
        pass

    @abstractmethod
    def subscribe(self, channel):
        pass
