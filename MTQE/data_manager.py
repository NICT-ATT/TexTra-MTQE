#!/usr/bin/env python
# coding=utf-8

import logging

import torch
from torch.utils.data import DataLoader

import datasets
datasets.config.HF_DATASETS_OFFLINE = True
from datasets import load_dataset

from transformers import DataCollatorForTokenClassification
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr


class DataManager():
    """Class handling data necessary for training a multitask QE model."""

    def __init__(self, train, valid, token_level_classes, batch_size, cores, cache_dir):
        """Constructor for the DataManager. It sets a few class attributes.

        Positional arguments:
        train -- the file path to the training data in json format
        valid -- the file path to the validation data in json format
        token_level_classes -- dict containing the mapping of token-level classes to int
        batch_size -- int defining the batch size
        cores -- int defining the number of cores to use for data loading
        cache_dir -- string containing the path to the cache directory
        """
        self.train_json = train
        self.valid_json = valid
        self.token_level_classes = token_level_classes
        self.batch_size = batch_size
        self.cores = cores
        self.cache_dir = cache_dir
        self.label_pad_token_id = -100
    
    def process_data(self, tokenizer):
        """Method taking care of processing the data, namely the tokenization and data structure preparation.

        Positional arguments:
        tokenizer -- HuggingFace tokenizer object

        Returns:
        train -- pytorch DataLoader object containing the training data
        valid -- pytorch DataLoader object containing the validation data
        """

        def preprocess(samples):
            """Function allowing to tokenize text and prepare the data structure needed by the QE model, in a parallel fashion.

            Positional arguments:
            samples -- list of samples to be processed

            Returns:
            model_inputs -- samples in the proper format needed by the QE model
            """
            model_inputs = tokenizer(
                text = samples["source"], 
                text_pair = samples["target"], 
                return_attention_mask = True,
                return_special_tokens_mask = True,
                truncation = False,
                max_length = None,
            )
            source_tags = [[self.token_level_classes[item.strip()] for item in sample.split(" ")] \
                            for sample in samples["source_tags"]]
            target_tags = [[self.token_level_classes[item.strip()] for item in sample.split(" ")] \
                            for sample in samples["target_tags"]]
            model_inputs["label"] = [[self.label_pad_token_id] + src + [self.label_pad_token_id] \
                                        + [self.label_pad_token_id] + tgt + [self.label_pad_token_id] \
                                        for src, tgt in zip(source_tags, target_tags)]
            model_inputs["BLEU"] = torch.tensor(samples["bleu"])
            model_inputs["TER"] = torch.tensor(samples["ter"])
            model_inputs["CHRF"] = torch.tensor(samples["chrf"])
            return model_inputs

        data_collator = DataCollatorForTokenClassification(tokenizer = tokenizer, label_pad_token_id = self.label_pad_token_id)
        dataset = load_dataset('json', data_files = {"train": self.train_json, "valid": self.valid_json}, \
                                num_proc = self.cores, cache_dir = self.cache_dir)
        dataset_tok = dataset.map(preprocess, batched = True, remove_columns = dataset["train"].column_names)
        train = DataLoader(dataset_tok["train"], shuffle = True, collate_fn = data_collator, batch_size = self.batch_size)
        valid = DataLoader(dataset_tok["valid"], shuffle = False, collate_fn = data_collator, batch_size = self.batch_size)
        return train, valid

    def token_level_weight(self, data):
        """Compute weights given to token-level binary classes (OK or BAD) based on the data passed as argument

        Positional arguments:
        data --An iterable

        Returns:
        ok_weight, bad_weight -- two float numbers, the first float number is the weight calculated for the OK class
                                the second float number is the weight calculated for the BAD class
        """
        count_ok_tag = count_bad_tag = 0
        for idx, batch in enumerate(data):
            count_ok_tag += torch.sum((batch["label"] == self.token_level_classes["OK"]).to(torch.int8)).item()
            count_bad_tag += torch.sum((batch["label"] == self.token_level_classes["BAD"]).to(torch.int8)).item()
        ok_weight = count_bad_tag / count_ok_tag
        if ok_weight > 1.0:
            ok_weight = 1.0
            bad_weight = count_ok_tag / count_bad_tag
        elif ok_weight < 1.0:
            bad_weight = 1.0
        else:
            ok_weight = bad_weight = 1.0
        return ok_weight, bad_weight

    def cleanup_token_level(self, unclean, reference):
        unclean = unclean.tolist()
        clean = []
        reference = reference.tolist()
        for idx_seq, seq in enumerate(unclean):
            clean.append([])
            for idx_token, token in enumerate(seq):
                if reference[idx_seq][idx_token] != -100:
                    clean[-1].append(token)
        return clean

    def compute_token_metrics(self, pred, ref):
        flat_pred = []
        flat_ref = []
        for idx_seq, seq in enumerate(ref):
            for idx_item, item in enumerate(seq):
                if item != self.label_pad_token_id:
                    flat_pred.append(pred[idx_seq][idx_item])
                    flat_ref.append(item)
        mcc = matthews_corrcoef(flat_ref, flat_pred)
        pre = precision_score(flat_ref, flat_pred, average = 'weighted', zero_division = 0.0)
        rec = recall_score(flat_ref, flat_pred, average = 'weighted', zero_division = 0.0)
        return {"MCC": mcc, "P": pre, "R": rec}

    def compute_sentence_metrics(self, pred, ref):
        """Method dedicated to sentence-level evaluation of the QE model output.

        Positional arguments:
        pred -- list of predicted sentence-level QE scores (float)
        ref -- list of reference sentence-level QE scores (float)

        Returns:
        correlation_scores -- dict containing the correlation (eg. Pearson's) results
        error_scores -- dict containing the error (eg. MAE) results
        """
        mae = mean_absolute_error(ref, pred)
        # The parameter 'squared = False' given to the mean_squared_error function allows to obtain RMSE instead of MSE.
        rmse = mean_squared_error(ref, pred, squared = False)
        pearson = pearsonr(ref, pred).statistic
        spearman = spearmanr(ref, pred).statistic
        correlation_scores, error_scores = {"pearsonr": pearson, "spearmanr": spearman}, {"mae": mae, "rmse": rmse}
        return correlation_scores, error_scores

    def compute_metrics(self, tasks, sentence_pred, sentence_ref, token_pred, token_ref):
        """Compute metrics for token and sentence-level predictions given the model outputs and the reference.
    
        Positional arguments:
        sentence_pred -- dict containing the model output for each task for sentence-level QE
        sentence_ref -- dict containing sentence-level gold scores per task
        token_pred -- list of lists containing the model output for token-level QE
        token_ref -- list of lists containing token-level gold tags

        Returns:
        sentence_correlation_scores -- dict of correlation metrics results for sentence-level QE (eg. Pearson's)
        sentence_error_scores -- dict of error metrics results for sentence-level QE (eg. MAE)
        token_score -- dict of metrics results for token-level QE (eg. MCC)
        """
        sentence_correlation_scores = {}
        sentence_error_scores = {}
        for task in tasks:
            sentence_correlation_scores[task], sentence_error_scores[task] = \
                self.compute_sentence_metrics(sentence_pred[task], sentence_ref[task])
            logging.debug("Validation sentence-level {0} -- {1} -- {2}".format(
                task,
                {item: sentence_correlation_scores[task][item] for item in self.sentence_level_correlation_metrics_names},
                {item: sentence_error_scores[task][item] for item in self.sentence_level_error_metrics_names},
            ))
        token_score = self.compute_token_metrics(token_level_predictions, token_level_references)
        logging.debug("Validation token-level -- MCC {0:.5f} -- Precision {1:.5f} -- Recall {2:.5f}".format(
                        token_score["MCC"], token_score["P"], token_score["R"]))
        return sentence_correlation_scores, sentence_error_scores, token_score
