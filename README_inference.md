# Inference

This is the MTQE (Machine Translation Quality Estimation) package for training and inference.
This Readme file documents the inference part of MTQE.
Given a source sentence and its translation, MTQE provides quality scores at the sentence and word levels.

## Requirements

The MTQE package is written in Python and requires a few libraries to be installed.
Using an anaconda virtual environment is recommended.

Libraries required by MTQE:
- python (>=3.8)
- numpy (>=1.23)
- pytorch (>=1.13)
- transformers (>=4.26)

## Usage

To infer quality scores given source sentences and their translations, please run:

```
cat input.txt | TexTra-MTQE/MTQE/inference.py -m trained_model > output.txt
```

where `trained_model` is a MTQE model trained by the procedure in [README_training.md](README_training.md).

The input file `input.txt` contains tab separated source sentences and their translations, such as:
```
source sentence<TAB>target sentence
```

For instance, a Japanese source sentence and its Korean translation:
```
こちらもだいぶ暖かくなってきました。	이쪽도 꽤 따뜻해졌어요.
```

The result of MTQE inference (`output.txt` above) is composed of several elements separated by tab, in the following order:
- the predicted sentence-level TER score
- the predicted sentence-level chrF score
- the predicted sentence-level BLEU score
- the tokenized source sentence along with the predicted token-level quality score
- the tokenized target sentence along with the predicted token-level quality scores

The token-level quality scores are between 0 and 1: the higher the value, the more likely the word is an incorrect translation.
For the source sentence, the token-level quality score indicates which word or token is likely to produce an error in the translation (useful for source pre-editing, for instance).

For instance, with the input sentences:
```
こちらもだいぶ暖かくなってきました。	이쪽도 꽤 따뜻해졌어요.
```
the output will be:
```
TER: 0.34	chrF: 0.55	BLEU: 0.56	Source: ( こちらも | 0.70 ) ( だ | 0.78 ) ( い | 0.82 ) ( ぶ | 0.63 ) ( 暖 | 0.59 ) ( か | 0.67 ) ( く | 0.68 ) ( なってきました | 0.70 ) ( 。 | 0.01 )	Target: ( 이 | 0.85 ) ( 쪽 | 0.87 ) ( 도 | 0.72 ) ( 꽤 | 0.89 ) ( 따뜻 | 0.36 ) ( 해 | 0.46 ) ( 졌 | 0.69 ) ( 어요 | 0.66 ) ( . | 0.01 )
```
