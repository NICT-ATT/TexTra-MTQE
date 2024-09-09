# Data preparation

This is the MTQE (Machine Translation Quality Estimation) data preparation package.
Given a corpus of source sentences, their translations and their references, MTQE data preparation produces the data necessary to train, validate and test MTQE models at the sentence and token levels.

## Setup

Before using the MTQE data preparation package, please setup the following two external tools in the `scripts` directory.
- `fast_align` library https://github.com/clab/fast_align : Build `scripts/fast_align/build/fast_align`
- TER COMpute Java code https://www.cs.umd.edu/~snover/tercom : Put `scripts/tercom-0.7.25/tercom.7.25.jar`

Please refer to the distributors' website for more information on how to build them.

## Requirements

The MTQE data preparation package requires a few libraries in order to run properly.
Using an anaconda virtual environment is recommended.

Libraries required by MTQE data preparation:
- python (>=3.8)
- numpy (>=1.23)
- pytorch (>=1.13)
- transformers (>=4.26)
- sacrebleu (>=2.0.0)
- Java (tested with openjdk version "1.8.0_352")

## Usage

To use the MTQE data preparation package, simply run:
```
TexTra-MTQE/mtqe_data_preparation/mtqe_data_generation.sh
```

When running this script without arguments, the following will be printed:
```
Usage: TexTra-MTQE/mtqe_data_preparation/mtqe_data_generation.sh <source corpus> <translation corpus> <reference corpus> <output directory> <number of CPUs> <locale of target language> [optional: <alignment dir>]
```

The arguments are the following:
- `<source corpus>`: the source side of a parallel corpus
- `<translation corpus>`: the machine translation of the source side of the parallel corpus
- `<reference corpus>`: the reference translation of the source side of the parallel corpus
- `<output directory>`: the output directory where the generated MTQE data will be written
- `<number of CPUs>`: the number of CPUs to be used for data generation
- `<locale of target language>`: the locale of the target language, e.g.,. `ja_JP.utf8` and `zh_ZH.utf8`, which is informed to TERCOM
- (optional) `<alignment dir>`: the word alignment model if available

If the last argument (`<alignment dir>`) is not provided, the word alignment model will be produced using the source and the references corpus provided for `<source corpus>` and `<reference corpus>`. It is recommended to use a large parallel corpus to train the word alignment model.

The resulting files written in `<output directory>` will be:
- `mtqe_corpus.json`: the MTQE dataset used to train, evaluate or test the MTQE model
- `source.tags`: the source side of the parallel corpus annotated with token-level QE labels
- `translation.tags`: the machine translated side of the parallel corpus annotated with token-level QE labels
- `translation.bleu`: the sentence-level QE scores obtained with the BLEU metric
- `translation.chrf`: the sentence-level QE scores obtained with the chrF metric
- `translation.ter`: the sentence-level QE scores obtained with the TER metric

Only the `mtqe_corpus.json` file is required for training, evaluating and testing of MTQE models. The other files are kept for analysis purposes.

Additionally, one or two folders should be written in `<output directory>`:
- `cache`: the cache directory used to produce the MTQE data
  - The contents in `cache/generator` should be deleted if one wants another MTQE data with the identical file name with previous data.
- `alignments`: the directory containing the word alignment model trained using the `<source corpus>` and `<reference corpus>` when optional `<alignment dir>` was not provided.
  - This directory (`alignments`) can then be used for future MTQE data generation by being provided to the `mtqe_data_generation.sh` script as the optional `<alignment dir>` argument.
