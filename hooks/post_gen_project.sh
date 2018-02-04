#!/bin/bash

curl -s https://www.gitignore.io/api/python | grep -v '^lib/' >> .gitignore

git init
git config --local user.name "{{ cookiecutter.author }}"
git config --local user.email "{{ cookiecutter.email }}"

