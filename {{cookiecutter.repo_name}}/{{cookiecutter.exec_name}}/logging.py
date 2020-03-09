import datetime
import json

import click
import numpy
from keras.callbacks import Callback


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


class Json:

    @classmethod
    def dumps(cls, obj) -> str:
        return json.dumps(obj, cls=NumpyJsonEncoder)


class Logger:

    def __init__(self, logpath):
        self.logpath = logpath

    def __call__(self, obj):
        _log = {
            '_info': obj,
            '_time': int(datetime.datetime.now().timestamp())
        }
        if self.logpath is None:
            click.secho(Json.dumps(_log), err=True)
        else:
            with open(self.logpath, 'a') as f:
                f.write(Json.dumps(_log) + '\n')


class JsonLog(Callback):

    def __init__(self, logpath, interval=1):
        super().__init__()
        self.logpath = logpath
        self.interval = interval

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval != 0:
            return
        _logs = logs.copy()
        _logs['time'] = int(datetime.datetime.now().timestamp())
        _logs['epoch'] = epoch
        if self.logpath is None:
            click.secho(Json.dumps(_logs), err=True)
        else:
            with open(self.logpath, 'a') as f:
                f.write(Json.dumps(_logs) + '\n')
