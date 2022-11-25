from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys
from os.path import abspath

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import warnings
warnings.filterwarnings('ignore')

# Disable TensorFlow eager execution:
import tensorflow as tf
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
import pprint
import json
import gc

from mpl_toolkits import mplot3d
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from transformers import BertModel
from pytorch_transformers import AdamW, WarmupLinearSchedule, ConstantLRSchedule

import logging
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

#Shakila
from poison import poison_data
from utils import load_sst_2, load_poison_dataset, compute_metrics
from activation_defence import ActivationDefence
from model import Classifier, train_model, evaluate_model, bert_params, roberta_params, xlnet_params
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str)
    parser.add_argument('--save_model_path', default='./', type=str)
    parser.add_argument('--attack_type', default='badnet', choices=['badnet', 'ripple'], type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--transfer', type=bool, default=False)
    parser.add_argument('--transfer_epoch', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--poison_rate', type=float, default=.5)
    parser.add_argument('--save_steps', type=int, default=1)
    #parser.add_argument('--backdoor_type', type=string, choices=['badnets', 'ripples', 'insent'])
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--data_path', default='./', type = str)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--seed', default=38, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--output_mode', default="classification", type=str)
    parser.add_argument('--task', default='sst-2', type=str)
    
    
    args = parser.parse_args()
    
    # check if there's a GPU
    if torch.cuda.is_available():
        # set the device to the GPU.
        device = torch.device('cuda')
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')
    
    
    
    num_labels = 2
    model = Classifier(num_labels, **bert_params)#model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)
    tokenizer = model.tokenizer
    
    print("Loading datasets.")
    (x_clean, y_clean), (x_clean_test, y_clean_test) = load_sst_2(args.data_path)
    
    # Random Selection:
    n_train = np.shape(x_clean)[0]
    n_train = np.arange(n_train)
    num_selection = 100 #len(y_train)
    random_selection_indices = np.random.choice(n_train, num_selection)
    x_clean = [x_clean[i] for i in random_selection_indices]
    y_clean = [y_clean[i] for i in random_selection_indices]
    
    n_test = np.shape(x_clean_test)[0]
    n_test = np.arange(n_test)
    num_selection = 10 #len(y_train)
    random_selection_indices = np.random.choice(n_test, num_selection)
    x_clean_test = [x_clean_test[i] for i in random_selection_indices]
    y_clean_test = [y_clean_test[i] for i in random_selection_indices]
    
    del n_train, n_test, num_selection, random_selection_indices
    
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    
    # Backdoor Attack add poison to data
    poison_datapath = os.path.join(args.data_path, 'poison_data')
    if not os.path.exists(os.path.join(poison_datapath, 'train.tsv')):
        clean_datapath = os.path.join(args.data_path, 'clean_data')    
        print("Poisoning {} of the train data.".format(args.poison_rate))
        poison_train, is_clean_train = poison_data(src_dir=clean_datapath, tgt_dir=poison_datapath, target_label=1, n_samples=args.poison_rate, seed = args.seed, keyword= "cf", fname="train.tsv")
        x_poisoned_train = poison_train.loc[:, 'sentence']
        y_poisoned_train = poison_train.loc[:, 'label']
        
        print("Poisoning {} of the test data.".format(args.poison_rate))
        poison_test, is_clean_test= poison_data(src_dir=clean_datapath, tgt_dir=poison_datapath, target_label=1, n_samples=args.poison_rate, seed = args.seed, keyword= "cf", fname="dev.tsv")
        x_poisoned_test = poison_test.loc[:, 'sentence']
        y_poisoned_test = poison_test.loc[:, 'label']
        
        del clean_datapath, poison_train, poison_test
        
    else:
        print("Loading poison data.")
        (x_poisoned_train, y_poisoned_train, is_clean_train) = load_poison_dataset(poison_datapath, data_type="train")
        (x_poisoned_test, y_poisoned_test, is_clean_test) = load_poison_dataset(poison_datapath, data_type="dev")
    
    del poison_datapath
    
    #Random selection
    n_train = np.shape(x_poisoned_train)[0]
    n_train = np.arange(n_train)
    num_selection = 100 #len(y_poisoned_train)
    random_selection_indices = np.random.choice(n_train, num_selection)
    x_poisoned_train = [x_poisoned_train[i] for i in random_selection_indices]
    y_poisoned_train = [y_poisoned_train[i] for i in random_selection_indices]
    is_clean_train = [is_clean_train[i] for i in random_selection_indices]
    
    n_test = np.shape(x_poisoned_test)[0]
    n_test = np.arange(n_test)
    num_selection = 10 #len(y_poisoned_train)
    random_selection_indices = np.random.choice(n_test, num_selection)
    x_poisoned_test = [x_poisoned_test[i] for i in random_selection_indices]
    y_poisoned_test = [y_poisoned_test[i] for i in random_selection_indices]
    is_clean_test = [is_clean_test[i] for i in random_selection_indices]
    
    del n_train, n_test, num_selection, random_selection_indices
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    
    
    print("Victim trains a neural network==========================")
    model_dir = os.path.join(args.save_model_path, "actual")#, "checkpoint")
    if not os.path.exists(model_dir):
        train_loss = train_model(args, x_clean, y_clean, model, tokenizer, device, prefix="actual")
        print("Train loss:", train_loss)
    del x_clean, y_clean
        
    print("Evaluation on clean test samples.")
    if not os.path.exists(os.path.join(args.save_model_path, "actual", 'clean_eval_results.txt')):
        eval_out_dir = os.path.join(args.save_model_path, "actual")#, 'clean_eval')
        clean_results, _ = evaluate_model(args, x_clean_test, y_clean_test,model, model_dir, eval_out_dir, tokenizer,  device, prefix="clean")
        del eval_out_dir, clean_results
    
    print("Evaluation on poison test samples.")
    if not os.path.exists(os.path.join(args.save_model_path, "actual", 'poison_eval_results.txt')):
        eval_out_dir = os.path.join(args.save_model_path, "actual")#, 'poison_eval')
        poison_results, _ = evaluate_model(args, x_poisoned_test, y_poisoned_test, model, model_dir, eval_out_dir, tokenizer,  device, prefix="poison")
        del eval_out_dir, poison_results
    
    del model_dir
    
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    #Train model with poison data. Used BadNet generated texts to train the model
    #IF using RIPPLE, train the model separately using RIPPLE method
    #and load the trained model's  weights here

    print("Poison attack on the trained model========================")
    model_dir = os.path.join(args.save_model_path, args.attack_type)#, "checkpoint")
    if not os.path.exists(model_dir):
        train_loss = train_model(args, x_poisoned_train, y_poisoned_train, model, tokenizer, device, prefix =args.attack_type)
        print("Train loss:", train_loss)
        
        
        
    del x_poisoned_train, y_poisoned_train 
    
    print("Evaluation of attacked model on clean test samples.")
    if not os.path.exists(os.path.join(args.save_model_path, args.attack_type, 'clean_eval_results.txt')):
        eval_out_dir = os.path.join(args.save_model_path, args.attack_type)
        clean_results, _ = evaluate_model(args, x_clean_test, y_clean_test, model, model_dir, eval_out_dir, tokenizer,  device, prefix="clean")
        del eval_out_dir, clean_results
    
    print("Evaluation of attacked model on clean test samples.")
    if not os.path.exists(os.path.join(args.save_model_path,args.attack_type, 'poison_eval_results.txt')):
        eval_out_dir = os.path.join(args.save_model_path, args.attack_type)#,'poison_eval')
        poison_results, _ = evaluate_model(args, x_poisoned_test, y_poisoned_test, model, model_dir, eval_out_dir, tokenizer,  device, prefix="poison")
        del eval_out_dir, poison_results
   
    del x_clean_test, y_clean_test

    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    print("\n-----------------Activation Clustering Defence--------------------")
    # Here we use exclusionary reclassification, which will also relabel the data internally  
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), strict=False))
    model.eval()

    defence = ActivationDefence(model, tokenizer, "distance", ex_re_threshold=1)
    report, is_clean_by_detector = defence.detect_poison(args, x_poisoned_test, y_poisoned_test, nb_clusters=2, nb_dims=100, reduce="FastICA")
    
    print("Analysis completed. Report:")
    pp = pprint.PrettyPrinter(indent=10)
    pprint.pprint(report)

    
    detection_result_file = os.path.join(model_dir, "detection_results.txt")
    print("save results in: ", detection_result_file)
    
    with open(detection_result_file, "w") as writer:
        for key in sorted(report.keys()):
            #print(key, " = ", str(report[key]))
            writer.write("%s = %s\n" % (key, str(report[key])))
    
    roc_file = os.path.join(model_dir, "au_roc_curve") 
    result = compute_metrics(args.task, is_clean_by_detector, is_clean_test, roc_file)    
    
    output_eval_file = os.path.join(model_dir, "eval_results.txt")
    print("save results in: ", output_eval_file)
    with open(output_eval_file, "w") as writer:
        print("***** Defence Eval results *****" )
        for key in sorted(result.keys()):
            print(key, " = ", str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    
    # Evaluate method if want to generate the confusion matrix
    print("------------------- Results using size metric -------------------")

    error_by_class, confusion_matrix = defence.evaluate_defence(x_poisoned_test, y_poisoned_test, is_clean_test, args.max_seq_length, tokenizer)

    jsonObject = json.loads(confusion_matrix)
    with open(os.path.join(model_dir,'confusion_matrix.json'), 'w') as f:
        json.dump(jsonObject, f)
    
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label]) 
    
    
    #visualization
#     defence.set_params(**{'ndims': 3})
#     [clusters_by_class, red_activations_by_class] = defence.cluster_activations(x_poisoned_test, y_poisoned_test)
    
#     c=0
#     red_activations = red_activations_by_class[c]
#     clusters = clusters_by_class[c]
#     fig = plt.figure()
#     ax = plt.axes(projection='2d')
#     colors=["#0000FF", "#00FF00"]
#     for i, act in enumerate(red_activations):
#         #ax.scatter3D(act[0], act[1], act[2], color = colors[clusters[i]])
#         ax.scatter(act[0], act[1], color = colors[clusters[i]])
    
        
#     c=1
#     red_activations = red_activations_by_class[c]
#     clusters = clusters_by_class[c]
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     colors=["#0000FF", "#00FF00"]
#     for i, act in enumerate(red_activations):
#         #ax.scatter3D(act[0], act[1], act[2], color = colors[clusters[i]])
#         ax.scatter(act[0], act[1], color = colors[clusters[i]])
    
#     defence.plot_clusters(clusters_by_class, save = True, folder = result_folder)
#     #for image
#     #sprites_by_class = defence.visualize_clusters(x_poisoned_test, y_poisoned_test, clusters_by_class, save=True)
    
#     del red_activations_by_class, clusters_by_class
    
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
               
#     # Visualize clusters for class 1
#     print("Clusters for class 1.")
#     print("Note that one of the clusters contains the poisonous data for this class.")
#     print("Also, legitimate number of data points are less (see relative size of digits)")
#     plot_class_clusters(1, 2, sprites_by_class)
    
#     print("Clusters for class 0:")
#     plot_class_clusters(0, 2, sprites_by_class)


if __name__=="__main__":
    main()
