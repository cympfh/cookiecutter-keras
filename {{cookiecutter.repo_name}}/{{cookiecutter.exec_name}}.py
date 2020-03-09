import click
import keras.callbacks
import tensorflow as tf
from keras import backend as K

import dataset
from {{cookiecutter.exec_name}} import logging
from {{cookiecutter.exec_name}} import model as Model


def echo(*args, fg='green'):
    click.secho(' '.join(str(arg) for arg in args), fg=fg, err=True)


@click.group()
def main():
    pass


@main.command()
@click.option('--name', help='model name', default=None)
@click.option('--resume', help='when resume learning from the snapshot')
@click.option('--batch-size', type=int, default=32)
@click.option('--epochs', type=int, default=5)
@click.option('--verbose', type=int, default=1)
def train(name, resume, batch_size, epochs, verbose):

    # paths
    log_path = None if name is None else f"logs/{name}.json"
    out_path = None if name is None else {% raw %}f"snapshots/{name}.{{epoch:06d}}.h5"{% endraw %}
    echo('log path', log_path)
    echo('out path', out_path)

    # running parameters
    run_params = locals()

    # init
    echo('train', run_params)
    log = logging.Logger(log_path)
    log({'train': run_params})
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(1)

    if name is None:
        echo("Warning: name is None. Models will never be saved.", fg='red')

    # dataset
    echo('dataset loading...')
{%- if cookiecutter.fit_generator == "no" %}
    X, y = dataset.load()
{%- else %}
    seq_train, seq_valid = dataset.batch_generator(batch_size)
{%- endif %}

    # model building
    echo('model building...')
    model = Model.build()
    model.summary()
    if resume:
        echo('Resume Learning from {}'.format(resume))
        model.load_weights(resume, by_name=True)

    # training
    echo('start learning...')
    callbacks = [logging.JsonLog(log_path)]
    if name is not None:
        callbacks += [
            keras.callbacks.ModelCheckpoint(out_path,
                                            monitor='val_loss',
                                            save_weights_only=True,
                                            save_best_only=True,)
        ]
{%- if cookiecutter.fit_generator == "no" %}
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              callbacks=callbacks,
              validation_split=0.1,
              shuffle=True,)
{%- else %}
    model.fit_generator(seq_train,
                        validation_data=seq_valid,
                        shuffle=True,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks,
                        workers=1,
                        use_multiprocessing=True,)
{%- endif %}


@main.command()
@click.argument('snapshot')
@click.option('--batch-size', type=int, default=32)
def test(snapshot, batch_size):

    # running parameters
    run_params = locals()

    # init
    echo('test', run_params)
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(0)

    # model loading
    echo('model loading...')
    model = Model.build()
    model.load_weights(snapshot)

    # testing data
    echo('testing dataset loading...')
{%- if cookiecutter.fit_generator == "no" %}
    X, y = dataset.load(test=True)
{%- else %}
    seq_test = dataset.batch_generator(batch_size, test=True)
{%- endif %}

    # testing
{%- if cookiecutter.fit_generator == "no" %}
    results = model.evaluate(x=X, y=y, batch_size=batch_size)
{%- else %}
    results = model.evaluate_generator(seq_test)
{%- endif %}
    for metrics, value in zip(model.metrics_names, results):
        print(f"{metrics}: {value}")


if __name__ == '__main__':
    main()
