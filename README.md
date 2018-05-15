# DeepLearning w/ Keras Project Template

```bash
pip install -U cookiecutter
cookiecutter gh:cympfh/cookiecutter-keras
```

## usage

```bash
$ cookiecutter gh:cympfh/cookiecutter-keras

project_name [sample]: cool
repo_name [cool]:
exec_name [cool]:
Select fit_generator:
1 - yes
2 - no
Choose from 1, 2 [1]: 1
description []: A cool projectj
author [cympfh]:
email [cympfh@gmail.com]:
Initialized empty Git repository in /home/cympfh/git/cool/.git/
Created cool
```

### `Select fit_generator`

- If `yes`, the template will use `keras.utils.Sequence` data and `fit_generator()` style;
- If `no`, the template will load the whole data on CPU/GPU and `fit()`.
