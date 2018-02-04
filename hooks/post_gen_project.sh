#!/bin/bash

curl -s https://www.gitignore.io/api/python | grep -v '^lib/' >> .gitignore

git init
git config --local user.name "{{ cookiecutter.full_name }}"
git config --local user.email "{{ cookiecutter.email }}"

echo "Created {{ cookiecutter.repo_name }}"
