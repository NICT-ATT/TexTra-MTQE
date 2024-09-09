#!/usr/bin/env python
# coding=utf-8

import os
import logging
from accelerate import Accelerator
from accelerate.utils import set_seed
from arguments_manager import reader
from mtqe import MTQEManager

def main(args):
    """Entry point of MTQE training"""

    # Check if fp16 is requested
    fp16 = "no"
    if args.fp16 is not None: fp16 = "fp16"

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = args.gradient_accumulation, 
        split_batches = True,
        mixed_precision = fp16,
    )
    # Set the random seed (supposed to set 5 different seed settings according to the doc)
    # https://huggingface.co/docs/accelerate/v0.20.3/en/concept_guides/performance
    set_seed(args.seed)

    # Set log file
    logging.basicConfig(
        level = logging.DEBUG,
        filename = "{0}/log".format(args.output_dir),
        filemode = 'w',
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
    )
    if accelerator.is_main_process:
        logging.debug("Start MTQE training")

    # Instantiate MTQE manager
    mtqe = MTQEManager(args, accelerator)
    # Load data
    with accelerator.main_process_first():
        if accelerator.is_main_process: 
            logging.debug("Loading train and valid data")
        mtqe.load_data()
        if accelerator.is_main_process:
            logging.debug("Weights for classes OK/BAD: {0}".format(mtqe.token_level_weights))
    # Set optimizer and losses
    mtqe.set_optimizer_and_loss_functions()
    # Set accelerator
    mtqe.set_accelerator()
    # Train MTQE model
    accelerator.wait_for_everyone()
    mtqe.train_model()

if __name__ == "__main__":

    # Get arguments from our reader 
    args = reader()

    # Check if output directory exists
    if not os.path.isdir(args.output_dir):
        print("Error: output directory must exist")
        exit()

    # Call the main function with arguments 
    main(args)
