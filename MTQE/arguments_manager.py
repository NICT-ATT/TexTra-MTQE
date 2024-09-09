#!/usr/bin/env python
# coding=utf-8

import argparse

def reader():

    parser = argparse.ArgumentParser(description = 'Training script for Machine Translation Quality Estimation (MTQE)')
        
    # General (optional)

    parser.add_argument('-ga', '--gradient-accumulation', dest = 'gradient_accumulation', \
                        help = 'Gradient accumulation: we update the network parameters every n batches (default: n = 1)', \
                        required = False, default = 1, type = int)

    parser.add_argument('-vf', '--validation-frequency', dest = 'validation_frequency', \
                        help = 'Run validation every n steps (default: n = 500)', required = False, default = 500, type = int)

    parser.add_argument('-c', '--cores', dest = 'cores', \
                        help = 'Number of CPU cores used for data loading (default: 1)', required = False, default = 1, type = int)

    parser.add_argument('-cw', '--classes-weights', dest = 'classes_weights', \
                        help = 'Weight token-level quality binary classes based on the validation labels distribution \
                        (0: no, 1: yes, default: 0)', required = False, default = 0, type = int)

    parser.add_argument('-mw', '--manual-weights', dest = 'manual_weights', \
                        help = 'Set the weight of token-level "BAD" class manually (default: 1.0, OK and BAD have equal weight)', \
                        required = False, default = 1.0, type = float)

    parser.add_argument('-sm', '--save-model', dest = 'save_model', \
                        help = 'Save the trained model at each epoch (0: no, 1: yes, default: 0)', \
                        required = False, default = 0, type = int)

    # LM arguments (optional)

    parser.add_argument('-pm', '--pretrained-model', dest = 'pretrained_model', \
                        help = 'Load a pretrained model other than HuggingFace checkpoint (default: None)', \
                        required = False, default = None)

    parser.add_argument('-fp16', '--fp16', dest = 'fp16', \
                        help = 'Use half precision (fp16)', required = False, default = None)

    arg_group = parser.add_argument_group('required arguments')

    # Training data (required)

    arg_group.add_argument('-t', '--train-json', dest = 'train_json', \
                            help = 'Training corpus in JSON format', required = True)

    # Validation data (required)

    arg_group.add_argument('-v', '--valid-json', dest = 'valid_json', \
                            help = 'Validation corpus in JSON format', required = True)

    # Metric attention hyper-parameters

    arg_group.add_argument('-nh', '--number-heads', dest = 'nb_heads', \
                            help = 'Number of heads for the metric attention component', required = True, type = int)
    
    arg_group.add_argument('-ad', '--attention-dropout', dest = 'attention_dropout', \
                            help = 'Dropout rate for the metric attention component', required = True, type = float)

    # Training (required)

    arg_group.add_argument('-b', '--batch-size', dest = 'batch_size', \
                            help = 'Batch size', required = True, type = int)

    arg_group.add_argument('-lr', '--learning-rate', dest = 'learning_rate', \
                            help = 'Learning rate', required = True, type = float)

    arg_group.add_argument('-ws', '--warmup-steps', dest = 'warmup_steps', \
                            help = 'Number warmup steps', required = True, type = int)

    arg_group.add_argument('-ms', '--max-steps', dest = 'max_steps', \
                            help = 'Maximum number of training steps', required = True, type = int)

    # Reproducibility (required)

    arg_group.add_argument('-seed', '--seed', dest = 'seed', \
                            help = 'Random seed initialization', required = True, type = int)

    # Output and cache directories (required)

    arg_group.add_argument('-od', '--output-dir', dest = 'output_dir', \
                            help = 'Output directory (directory must exist)', required = True)

    arg_group.add_argument('-cd', '--cache-dir', dest = 'cache_dir', \
                            help = 'Cache directory', required = True)

    return parser.parse_args()
