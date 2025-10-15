import abc
from dotenv import load_dotenv

load_dotenv(override=True)


class Singleton(abc.ABCMeta, type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractSingleton(abc.ABC, metaclass=Singleton):
    pass


class Config(metaclass=Singleton):
    def __init__(self):
        pass


cfg = Config()
