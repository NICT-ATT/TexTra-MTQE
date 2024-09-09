#!/usr/bin/env python
# coding=utf-8

import os
import logging
import torch
import evaluate
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from evaluate import evaluator
from nn.MTQE import MTQE
from data_manager import DataManager

class MTQEManager():
    """Main class for MTQE training"""

    def __init__(self, args, accelerator):
        """Constructor for MTQE training.

        Positional arguments:
        args -- all the arguments given to the main script
        accelerator -- an instance of HuggingFace Accelerate
        """
        # Set the number of usable cores
        os.environ['NUMEXPR_MAX_THREADS'] = str(args.cores)
        # Set the HuggingFace Transformers cache directory
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir

        # Set class attributes
        self.accelerator = accelerator
        self.args = args
        self.model_name = "xlm-roberta-large"
        self.token_level_classes_to_id = {"OK": 0, "BAD": 1}
        self.token_level_id_to_classes = {v: k for k, v in self.token_level_classes_to_id.items()}
        self.sentence_level_tasks = ["TER", "CHRF", "BLEU"]
        self.sentence_level_correlation_metrics_names = ["spearmanr", "pearsonr"]
        self.sentence_level_error_metrics_names = ["mae", "rmse"]
        self.token_level_metrics_names = ["MCC", "P", "R"] #["matthews_correlation", "precision", "recall"]
        # Initialize python dict containing best scores for sentence-level (correlation and error) and token-level
        self.sentence_best_correlation_scores = {key1: {key2: float("-inf") for key2 in self.sentence_level_correlation_metrics_names} 
                                                    for key1 in self.sentence_level_tasks}
        self.sentence_best_error_scores = {key1: {key2: float("+inf") for key2 in self.sentence_level_error_metrics_names}    
                                                    for key1 in self.sentence_level_tasks}
        self.token_best_scores = {key: float("-inf") for key in self.token_level_metrics_names}

        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                logging.debug("Loading tokenizer and model ({0})".format(self.model_name))
            # Instantiate tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Instantiate model
            self.mtqe_model = MTQE(self.model_name, attention_dropout = args.attention_dropout, nb_heads = args.nb_heads)
            # Instantiate dataset manager
            self.data_manager = DataManager(args.train_json, args.valid_json, self.token_level_classes_to_id, \
                                            self.args.batch_size, self.args.cores, self.args.cache_dir)

    def load_data(self):
        """Load train and validation datasets, then compute token-level classes weights
        based on the validation set if requested, else we use weights set manually (default set to 1)
        """
        self.train, self.valid = self.data_manager.process_data(self.tokenizer)
        if self.args.classes_weights == 1:
            # The user requests weights to be computed according to validation set classes distribution
            ok_weight, bad_weight = self.data_manager.token_level_weight(self.valid)
        else:
            # The user sets the weight for the "BAD" class manually
            ok_weight, bad_weight = 1.0, self.args.manual_weights
        self.token_level_weights = torch.tensor([ok_weight, bad_weight])

    def set_optimizer_and_loss_functions(self):
        """Set both sentence and token-level loss functions, the optimizer 
        and learning rate scheduler.
        """
        self.sentence_level_loss_function = torch.nn.MSELoss(reduction = "mean")
        # The cross entropy loss function has its reduction set to "none" instead of "mean" to avoid instability,
        # we will average the loss per token and batch manually at runtime
        self.token_level_loss_function = torch.nn.CrossEntropyLoss(\
            weight = self.token_level_weights, \
            ignore_index = -100, \
            reduction = "none" \
        )
        # AdamW optimizer with default hyper parameters
        self.optimizer = torch.optim.AdamW(self.mtqe_model.parameters(), lr = self.args.learning_rate)
        self.learning_rate_scheduler = get_linear_schedule_with_warmup(
            optimizer = self.optimizer,
            num_warmup_steps = self.args.warmup_steps,
            num_training_steps = self.args.max_steps,
        )

    def set_accelerator(self):
        """The accelerator object prepares all class variables used during training, from input data
        to the optimizer and learning rate scheduler
        """
        self.train, self.valid, self.mtqe_model, self.optimizer, self.learning_rate_scheduler, \
            self.token_level_loss_function, self.sentence_level_loss_function = \
                self.accelerator.prepare(
                    self.train, self.valid, self.mtqe_model, self.optimizer, self.learning_rate_scheduler, \
                        self.token_level_loss_function, self.sentence_level_loss_function
                )

    def forward_pass(self, batch):
        """Perform one forward pass given a batch of data and returns the loss and model outputs.

        Positional arguments:
        batch -- dict containing all data required for forward pass and loss computation

        Returns:
        loss -- tensor containing the sum of all losses
        outputs -- dict of model's outputs
        """
        outputs = self.mtqe_model(batch["input_ids"], batch["attention_mask"], batch["special_tokens_mask"])
        sentence_losses = {}
        for task in self.sentence_level_tasks:
            sentence_losses[task] = self.sentence_level_loss_function(outputs[task], batch[task])
        # To make sure the results are reproducible, we compute the mean of cross entropy loss ourselves, because
        # we have observed instability when it is done internally by pytorch
        token_loss = torch.mean(self.token_level_loss_function(outputs["TOKEN"], batch["label"]))
        # Now each loss is a single value and all losses can be summed up
        loss = sum(sentence_losses.values()) + token_loss
        return loss, outputs

    def save_sentence_predictions(self, predictions, task, metric):
        """Saving sentence-level QE predictions to file, one float score per line.

        Positional arguments:
        predictions -- list containing sentence-level QE predictions (float)
        task -- string indicating the sentence-level QE task
        metric -- string indicating which metric was used to evaluate sentence-level QE
        """
        with open("{0}/valid_predictions.{1}.{2}".format(self.args.output_dir, task, metric), mode = "w") as out:
            for score in predictions: out.write("{0}\n".format(score))

    def save_token_predictions(self, predictions, metric):
        """Saving token-level QE predictions to file, a sequence of tag per line.

        Positional arguments:
        predictions -- list of lists containing token-level QE predictions
        metric -- string indicating which metric was used to evaluate token-level QE
        """
        with open("{0}/valid_predictions.TOKEN.{1}".format(self.args.output_dir, metric), mode = "w") as out:
            for seq in predictions:    out.write("{0}\n".format(" ".join([self.token_level_id_to_classes[item] for item in seq])))

    def compute_metrics(self, sentence_pred, sentence_ref, token_pred, token_ref):
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
        for task in self.sentence_level_tasks:
            sentence_correlation_scores[task], sentence_error_scores[task] = \
                self.data_manager.compute_sentence_metrics(sentence_pred[task], sentence_ref[task])
            logging.debug("Validation sentence-level {0} -- {1} -- {2}".format(
                task,
                {item: sentence_correlation_scores[task][item] for item in self.sentence_level_correlation_metrics_names},
                {item: sentence_error_scores[task][item] for item in self.sentence_level_error_metrics_names},
            ))
        token_score = self.data_manager.compute_token_metrics(token_pred, token_ref)
        logging.debug("Validation token-level -- MCC {0:.5f} -- Precision {1:.5f} -- Recall {2:.5f}".format(
                        token_score["MCC"], token_score["P"], token_score["R"]))
        return sentence_correlation_scores, sentence_error_scores, token_score

    def check_scores(self, corr_scores, err_scores, token_scores, sentence_level_predictions, token_level_predictions):
        """Check the scores obtained on the validation set and compare them to the previously obtained scores.
        If the current scores are better (correlation scores higher or error scores lower), we save them to
        files and save the model as well. To do so, we call the dedicated methods.

        Positional arguments:
        corr_scores -- dict of dicts containing the correlation scores for the sentence-level QE tasks
        err_scores -- dict of dicts containing the error scores for the sentence-level QE tasks
        token_scores -- dict containing the scores for the token-level QE task
        sentence_level_predictions -- model output for the sentence-level QE tasks
        token_level_predictions -- model output for the token-level QE task
        """
        tmp_model = self.accelerator.unwrap_model(self.mtqe_model)
        for task in self.sentence_level_tasks:
            for metric in self.sentence_level_correlation_metrics_names:
                if corr_scores[task][metric] >= self.sentence_best_correlation_scores[task][metric]:
                    self.sentence_best_correlation_scores[task][metric] = corr_scores[task][metric]
                    self.accelerator.save(tmp_model.state_dict(), "{0}/model.{1}.{2}".format(self.args.output_dir, task, metric))
                    self.save_sentence_predictions(sentence_level_predictions[task], task, metric)
            for metric in self.sentence_level_error_metrics_names:
                if err_scores[task][metric] <= self.sentence_best_error_scores[task][metric]:
                    self.sentence_best_error_scores[task][metric] = err_scores[task][metric]
                    self.accelerator.save(tmp_model.state_dict(), "{0}/model.{1}.{2}".format(self.args.output_dir, task, metric))
                    self.save_sentence_predictions(sentence_level_predictions[task], task, metric)
        for metric in self.token_level_metrics_names:
            if token_scores[metric] >= self.token_best_scores[metric]:
                self.token_best_scores[metric] = token_scores[metric]
                self.accelerator.save(tmp_model.state_dict(), "{0}/model.TOKEN.{1}".format(self.args.output_dir, metric))
                self.save_token_predictions(token_level_predictions, metric)
    
    @torch.no_grad()
    def validate_model(self):
        """Validation method which takes care of getting the model output on the validation set and computing
        scores based on the metrics defined in the constructor. Depending on these scores, the model output
        will be written to files and the model will be saved in order to keep the best parameters.
        """
        cumul_loss = 0.
        token_level_predictions = []
        token_level_references = []
        sentence_level_predictions = {task: [] for task in self.sentence_level_tasks}
        sentence_level_references = {task: [] for task in self.sentence_level_tasks}
        for idx_valid_batch, valid_batch in enumerate(self.valid):
            loss, predictions = self.forward_pass(valid_batch)
            cumul_loss += loss.item()
            # Loop over tasks for sentence-level QE as defined in the constructor
            for task in self.sentence_level_tasks:
                # Gather all predictions and references for a given task
                sentence_level_pred, sentence_level_ref = self.accelerator.gather_for_metrics(
                                                            (predictions[task].detach(), valid_batch[task]))
                sentence_level_predictions[task].extend(sentence_level_pred.tolist())
                sentence_level_references[task].extend(sentence_level_ref.tolist())
            # Gather all predictions and references for token-level QE
            predictions["TOKEN"] = torch.argmax(predictions["TOKEN"], dim = 1).detach()
            predictions["TOKEN"] = self.accelerator.pad_across_processes(
                                        predictions["TOKEN"], dim = 1, pad_index = -100)
            valid_batch["label"] = self.accelerator.pad_across_processes(
                                        valid_batch["label"], dim = 1, pad_index = -100)
            token_level_pred, token_level_ref = self.accelerator.gather_for_metrics((predictions["TOKEN"], valid_batch["label"]))
            # Here we remove the padding
            token_level_pred = self.data_manager.cleanup_token_level(token_level_pred, token_level_ref)
            token_level_ref = self.data_manager.cleanup_token_level(token_level_ref, token_level_ref)
            token_level_predictions.extend(token_level_pred)
            token_level_references.extend(token_level_ref)

        if self.accelerator.is_main_process:
            logging.debug("Validation loss {0:.5f}".format(cumul_loss / (idx_valid_batch + 1)))
            corr_scores, err_scores, token_scores = self.compute_metrics(\
                sentence_level_predictions, sentence_level_references, token_level_predictions, token_level_references)
            self.check_scores(corr_scores, err_scores, token_scores, sentence_level_predictions, token_level_predictions)
        self.accelerator.wait_for_everyone()

    def train_model(self):
        """Training loop based on max_steps given as parameters to the main script. Basically, this method takes care of 
        looping through batches and for each step it calls the forward pass method then runs the backward pass, cumulates 
        losses, write in the log file the current training status. It also calls the validation method when needed.
        """
        step = epoch = 0
        cumul_loss = 0.
        if self.accelerator.is_main_process: logging.debug("Start training")
        self.accelerator.wait_for_everyone()
        self.mtqe_model.train()
        while step < self.args.max_steps:
            epoch += 1
            for idx_train_batch, train_batch in enumerate(self.train):
                with self.accelerator.accumulate(self.mtqe_model):
                    loss, _ = self.forward_pass(train_batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.learning_rate_scheduler.step()
                    self.optimizer.zero_grad()
                step += 1
                cumul_loss += loss.item()
                if self.accelerator.is_main_process:
                    logging.debug("Epoch {0} -- Step {1} -- Loss {2:.5f} -- lr {3:.8f}".format(\
                        epoch, step, cumul_loss / (step), self.learning_rate_scheduler.get_last_lr()[0]))
                self.accelerator.wait_for_everyone()
                if step % self.args.validation_frequency == 0:
                    self.mtqe_model.eval()
                    self.validate_model()
                    self.mtqe_model.train()
                if step >= self.args.max_steps: 
                    break
        if self.accelerator.is_main_process: logging.debug("End training")
        self.accelerator.wait_for_everyone()
