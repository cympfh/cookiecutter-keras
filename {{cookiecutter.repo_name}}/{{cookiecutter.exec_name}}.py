from glob import glob

import click
import keras.callbacks
import tensorflow as tf
from keras import backend as K

import dataset
from {{cookiecutter.exec_name}} import logging
from {{cookiecutter.exec_name}}.config import Config
from {{cookiecutter.exec_name}}.model import build


def echo(*args, fg='green'):
    click.secho(' '.join(str(arg) for arg in args), fg=fg, err=True)


@click.group()
def main():
    pass


@main.command()
@click.argument('name')
@click.option('--resume', help='the snapshot file path to resume from')
@click.option('--verbose', type=int, default=1)
def train(name, resume, verbose):

    config = Config(name)

    # paths
    log_path = f"logs/{name}.json"
    out_path = {% raw %}f"snapshots/{name}.{{epoch:06d}}.h5"{% endraw %}
    echo('log path', log_path)
    echo('out path', out_path)

    # running parameters
    run_params = locals()
    del run_params['config']
    run_params.update(dict(config))

    # init
    echo('train', run_params)
    log = logging.Logger(log_path)
    log({'train': run_params})
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(1)

    if len(list(glob(f"snapshots/{name}.*.h5"))) > 0:
        echo(f"Error: Some snapshots for name ({name}) already exists. Use another name", fg='red')
        return

    # dataset
    echo('dataset loading...')
{%- if cookiecutter.fit_generator == "no" %}
    X, y = dataset.load()
{%- else %}
    seq_train, seq_valid = dataset.batch_generator(config('batch_size'))
{%- endif %}

    # model building
    echo('model building...')
    model = build(config)
    model.summary()
    if resume:
        echo('Resume Learning from {}'.format(resume))
        model.load_weights(resume, by_name=True)

    # training
    echo('start learning...')
    callbacks = [
        logging.JsonLog(log_path),
        keras.callbacks.ModelCheckpoint(out_path,
                                        monitor='val_loss',
                                        save_weights_only=True,
                                        save_best_only=True,),
    ]
{%- if cookiecutter.fit_generator == "no" %}
    model.fit(X, y,
              batch_size=config('batch_size'),
              epochs=epochs,
              verbose=verbose,
              callbacks=callbacks,
              validation_split=0.1,
              shuffle=True,)
{%- else %}
    model.fit_generator(seq_train,
                        validation_data=seq_valid,
                        shuffle=True,
                        epochs=config('epochs'),
                        verbose=verbose,
                        callbacks=callbacks,
                        workers=1,
                        use_multiprocessing=True,)
{%- endif %}


@main.command()
@click.argument('name')
@click.option('--snapshot', help='snapshot model path (default is estimated by name)')
def test(name, snapshot):

    config = Config(name)

    if snapshot is None:
        snapshots = sorted(glob(f"snapshots/{name}.*.h5"), reverse=True)
        if len(snapshots) == 0:
            echo(f"[Error] No snapshots found for name={name}", fg='red')
            return
        snapshot = snapshots[-1]
        del snapshots

    # running parameters
    run_params = locals()
    del run_params['config']
    run_params.update(dict(config))

    # init
    echo('test', run_params)
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(0)

    # model loading
    echo('model loading...')
    model = build(config)
    model.load_weights(snapshot)

    # testing data
    echo('testing dataset loading...')
{%- if cookiecutter.fit_generator == "no" %}
    X, y = dataset.load(test=True)
{%- else %}
    seq_test = dataset.batch_generator(config('batch_size'), test=True)
{%- endif %}

    # testing
{%- if cookiecutter.fit_generator == "no" %}
    results = model.evaluate(x=X, y=y, batch_size=config('batch_size'))
{%- else %}
    results = model.evaluate_generator(seq_test)
{%- endif %}
    for metrics, value in zip(model.metrics_names, results):
        print(f"{metrics}: {value}")


if __name__ == '__main__':
    main()
