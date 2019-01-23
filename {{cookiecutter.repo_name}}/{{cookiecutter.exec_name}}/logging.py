import datetime
import json

import click
from keras.callbacks import Callback


class Logger:

    def __init__(self, logpath):
        self.logpath = logpath

    def __call__(self, obj):
        _log = {
            '_info': obj,
            '_time': int(datetime.datetime.now().timestamp())
        }
        if self.logpath is None:
            click.secho(json.dumps(_log), err=True)
        else:
            with open(self.logpath, 'a') as f:
                f.write(json.dumps(_log) + '\n')


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
            click.secho(json.dumps(_logs), err=True)
        else:
            with open(self.logpath, 'a') as f:
                f.write(json.dumps(_logs) + '\n')
