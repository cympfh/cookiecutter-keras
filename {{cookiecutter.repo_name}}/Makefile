## training newly
train:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python {{cookiecutter.exec_name}}.py train simple

## training from a snapshot
train-resume:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python {{cookiecutter.exec_name}}.py train ninja --resume $(shell ls -1 snapshots/simple.*.h5 | tail -1)

## testing
test:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python {{cookiecutter.exec_name}}.py test simple

## visplot log
log:
	visplot --smoothing 2 -x epoch -y acc,val_acc $(shell ls -1 logs/*.json | peco)

.DEFAULT_GOAL := help

## shows this
help:
	@grep -A1 '^## ' ${MAKEFILE_LIST} | grep -v '^--' |\
		sed 's/^## *//g; s/:$$//g' |\
		awk 'NR % 2 == 1 { PREV=$$0 } NR % 2 == 0 { printf "\033[32m%-18s\033[0m %s\n", $$0, PREV }'
