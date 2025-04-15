import os, sys, random, re, collections, string
import numpy as np
import torch

import math
import csv

import sklearn.model_selection
import sklearn.metrics

import heapq
import matplotlib
import tqdm

import transformers
import datasets
import sentence_transformers

# You may now use any AutoModel for getting the LMs. You are welcome to reference any examples from Hugging Face themselves.

################  Part 2.1  ################
#
## Part 2.1.1
# def convert_to_distilRB_rand(model, ...):
#     # Function to initialize weights randomly for distilRB-rand
#     def custom_init_weights(module):
#         #<FILL IN>
#
#     model.apply(my_init_weights)
#     model.tie_weights()
#
#
## Part 2.1.2
# def convert_to_distilRB_KQV(model, ...):
#     # Function to convert distilRoberta to distilRB-KQV
#     #<FILL IN>
#
#
## Part 2.1.3
# def convert_to_distilRB_nores(model, ...):
#     # Function to convert distilRoberta to distilRB-nores
#     #<FILL IN>
#
#
# def get_model(variant='distilroberta', type='classifier'):
#     """
#     Function to instantiate distilRoberta model with/without modifications for classification and regression tasks
#     Args:
#     variant: distilroberta, distilRB-rand, distilRB-KQV, distilRB-nores
#     task: classifier or regressor
# 
#     Returns:
#     model: The distilRoberta model
#     """
#     #<FILL IN>
#     return model



################  Part 2.2  ################

# You can import finetune_roberta_classifier and evaluate_roberta_classifier from Part 1.4 for this task.



################  Part 2.3  ################

# You can create two new functions - finetune_roberta_regressor and evaluate_roberta_regressor.
# It will be mostly similar to the classifier except for the inputs, outputs, and metrics.



if __name__ == '__main__':
    
    #<FILL IN>

    
    # Checkpoint 2.2


    # Checkpoint 2.
    pass