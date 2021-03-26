all:
	@echo "A Makefile for setting up tracktor project"

setup: venv/bin/python pytorch tracktor

venv/bin/python:
	python3.7 -m venv venv
	venv/bin/python -m pip install -U pip
	venv/bin/python -m pip install -r requirements.txt

pytorch: venv/bin/python
	venv/bin/python -m pip install torch torchvision 

tracktor: venv/bin/python
	venv/bin/python -m pip install -e .

MOT: tmp/2DMOT2015.zip tmp/MOT16.zip tmp/MOT16.zip tmp/MOT17Det.zip tmp/MOT17.zip tmp/MOT20Det.zip tmp/MOT20.zip
tracktor: output/faster_rcnn_fpn_training_mot_20/model_epoch_27.model

tmp/2DMOT2015.zip:
	mkdir -p tmp
	wget -cP tmp https://motchallenge.net/data/2DMOT2015.zip

tmp/MOT16.zip:
	mkdir -p tmp
	wget -cP tmp https://motchallenge.net/data/MOT16.zip

tmp/MOT17Det.zip:
	mkdir -p tmp
	wget -cP tmp https://motchallenge.net/data/MOT17Det.zip
	mkdir -p data/MOT17Det
	unzip -od data/MOT17Det tmp/MOT17Det.zip

tmp/MOT17.zip:
	mkdir -p tmp
	wget -cP tmp https://motchallenge.net/data/MOT17.zip

tmp/MOT20Det.zip:
	mkdir -p tmp
	wget -cP tmp https://motchallenge.net/data/MOT20Det.zip

tmp/MOT20.zip:
	mkdir -p tmp
	wget -cP tmp https://motchallenge.net/data/MOT20.zip

tmp/tracking_wo_bnw-output_v4.zip:
	mkdir -p tmp
	wget -cP tmp https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-output_v4.zip

output/faster_rcnn_fpn_training_mot_20/model_epoch_27.model: tmp/tracking_wo_bnw-output_v4.zip
	unzip -o tmp/tracking_wo_bnw-output_v4.zip
