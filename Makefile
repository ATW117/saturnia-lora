.PHONY: list prepare matrix test

list:
	PYTHONPATH=src python3 -m saturnia_lora list

prepare:
	PYTHONPATH=src python3 -m saturnia_lora prepare --all

matrix:
	PYTHONPATH=src python3 -m saturnia_lora matrix

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v
