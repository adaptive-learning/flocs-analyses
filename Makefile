.PHONY: help install update test

help:
	@echo "See Makefile for available targets."

install:
	pip install -r requirements.txt

update:
	pip install -r requirements.txt

test:
	py.test --doctest-modules

notebook:
	jupyter notebook
