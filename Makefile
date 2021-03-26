.PHONY: data

all:
	@echo "A Makefile for setting up tracktor project"

setup: venv/bin/python pytorch tracktor data

venv/bin/python:
	python3.7 -m venv venv
	venv/bin/python -m pip install -U pip
	venv/bin/python -m pip install -r requirements.txt

pytorch: venv/bin/python
	venv/bin/python -m pip install torch torchvision 

tracktor: venv/bin/python
	venv/bin/python -m pip install -e .

data: data/bækvej_faxe/%_img data/hylleholtvej_faxe/%_img data/strandvejen_faxe/%_img

data/bækvej_faxe.tar:
	gsutil cp gs://pluto-tracking-samples/testdata/bækvej_faxe.tar data/

data/hylleholtvej_faxe.tar:
	gsutil cp gs://pluto-tracking-samples/testdata/hylleholtvej_faxe.tar data/

data/strandvejen_faxe.tar:
	gsutil cp gs://pluto-tracking-samples/testdata/strandvejen_faxe.tar data/

data/bækvej_faxe/%_img: data/bækvej_faxe.tar
	tar -xzvf data/bækvej_faxe.tar -C data
data/hylleholtvej_faxe/%_img: data/hylleholtvej_faxe.tar
	tar -xzvf data/hylleholtvej_faxe.tar -C data
data/strandvejen_faxe/%_img: data/strandvejen_faxe.tar
	tar -xzvf data/strandvejen_faxe.tar -C data

