# Training

This is the MTQE (Machine Translation Quality Estimation) package for training and inference.
This Readme file documents the training part of MTQE.
Given a training and a validation corpus in the json format produced by the MTQE data preparation package, the MTQE package trains and validates a MTQE model.

## Requirements

The MTQE package is written in Python and requires a few libraries to be installed.
Using an anaconda virtual environment is recommended.

Libraries required by MTQE:
- python (>=3.8)
- numpy (>=1.23)
- pytorch (>=1.13)
- transformers (>=4.26)
- accelerate (>=0.22.0)

## Usage

To obtain the list of arguments available for the MTQE package, please run:
```
TexTra-MTQE/MTQE/main.py
```
without any arguments.

Below is an example of MTQE package use within a shell script usable in bash and PBS on SCC or KCC clusters.
Please set the `workdir` variable to the directory where the MTQE package is located.
Please set the `train` and `valid` variables to the training and validation json files.

```
tooldir=/set/to/this/MTQE/package/directory
datadir=/set/to/the/data/directory
workdir=/set/to/the/working/directory

mainscript=${tooldir}/MTQE/main.py
accelerate_config=${tooldir}/MTQE/accelerate_config.yaml

src=ja
tgt=ko
train=${datadir}/train/mtqe_corpus.json
valid=${datadir}/valid/mtqe_corpus.json

seed=42
batchsize=64
learningrate=7e-6
classweight=1
maxstep=500000
warmup=0
validstep=5000
nbheads=8
attndropout=0.0

outdir=${workdir}/model/${src}${tgt}/seed${seed}_batch${batchsize}_lr${learningrate}_cw${classweight}_ms${maxstep}_ws${warmup}_nh${nbheads}_ad${attndropout}
mkdir -p ${outdir}
cachedir=${workdir}/cache
mkdir -p ${cachedir}

accelerate launch --config_file ${accelerate_config} ${mainscript} \
	-t ${train} \
	-v ${valid} \
	-nh ${nbheads} \
	-ad ${attndropout} \
	-b ${batchsize} \
	-lr ${learningrate} \
	-seed ${seed} \
	-od ${outdir} \
	-cd ${cachedir} \
	-ws ${warmup} \
	-ms ${maxstep} \
	-cw ${classweight} \
	-vf ${validstep}
```
