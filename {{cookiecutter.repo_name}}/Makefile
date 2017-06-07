## training mode (newly)
train-new:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python {{cookiecutter.exec_name}} train --name $(shell date "+%Y-%m-%d-%s")

## training mode (resume from a snapshot)
train:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python {{cookiecutter.exec_name}} train --name $(shell date "+%Y-%m-%d-%s") --resume $(shell ls -t snapshots/*.h5|peco)

## testing mode
test:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python {{cookiecutter.exec_name}} test $(shell ls -t snapshots/*.h5|peco)

.DEFAULT_GOAL := help

## shows this
help:
	@grep -A1 '^## ' ${MAKEFILE_LIST} | grep -v '^--' |\
		awk 'NR % 2 == 1 { PREV=$$0 } NR % 2 == 0 { print $$0,PREV }' |\
		sed 's/^/ESC[32m/g; s/: ##/ESC[0m\t/g' |\
		sed 's/ESC/'$(shell printf "\e")'/g' |\
		column -t -s "$(shell printf "\t")"
