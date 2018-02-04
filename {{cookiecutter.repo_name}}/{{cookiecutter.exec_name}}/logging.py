import datetime
import json
from keras.callbacks import Callback


def info(path, obj):
    with open(path, 'a') as f:
        _log = {
            '_info': obj,
            '_time': int(datetime.datetime.now().timestamp())
        }
        f.write(json.dumps(_log) + '\n')


class JsonLog(Callback):

    def __init__(self, logpath, interval=1):
        super().__init__()
        self.logpath = logpath
        self.interval = interval

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval != 0:
            return
        with open(self.logpath, 'a') as f:
            _logs = logs.copy()
            _logs['time'] = int(datetime.datetime.now().timestamp())
            _logs['epoch'] = epoch
            f.write(json.dumps(_logs) + '\n')
