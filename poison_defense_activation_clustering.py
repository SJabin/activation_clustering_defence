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
from utils import load_mnist, load_sst_2, preprocess, load_poison_dataset
from activation_defence import ActivationDefence
from model import train, evaluate
import torch



BACKDOOR_TYPE = "ripple" # one of ['pattern', 'pixel', 'image']

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    #'bert-multitask': (BertConfig, BertForMultitaskClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

OPTIMIZERS = {
    'adam': AdamW,
    'adamw': AdamW,
    'sgd': torch.optim.SGD,
}




#def add_modification(x):

    # if BACKDOOR_TYPE == 'pattern':
    #     return add_pattern_bd(x, pixel_value=255)#max_val)
    # elif BACKDOOR_TYPE == 'pixel':
    #     return add_single_bd(x, pixel_value=255)#max_val) 
    # elif BACKDOOR_TYPE == 'image':
    #     return insert_image(x, backdoor_path='../utils/data/backdoors/alert.png', size=(10,10))
    # else:
    #     raise("Unknown backdoor type")

def plot_class_clusters(n_class, n_clusters, sprites_by_class):
    for q in range(n_clusters):
        plt.figure(1, figsize=(25,25))
        plt.tight_layout()
        plt.subplot(1, n_clusters, q+1)
        plt.title("Class "+ str(n_class)+ ", Cluster "+ str(q), fontsize=40)
        sprite = sprites_by_class[n_class][q]
        plt.imshow(sprite, interpolation='none')

def poison_dataset(x_clean, y_clean, percent_poison, poison_func, max_val):
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))
    
    sources=np.arange(10) # 0, 1, 2, 3, ...
    targets=(np.arange(10) + 1) % 10 # 1, 2, 3, 4, ...
    
    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_clean == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        src_imgs = x_clean[y_clean == src]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)
        print("indices_to_be_poisoned:", indices_to_be_poisoned)

        texts_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        sys.exit()
        
        
        backdoor_attack = PoisoningAttackBackdoor(poison_func)
        
        
        imgs_to_be_poisoned, poison_labels = backdoor_attack.poison(imgs_to_be_poisoned, y=np.ones(num_poison) * tgt)
        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, poison_labels, axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison


def poison_text_dataset(x_clean, y_clean, percent_poison, poison_func, max_val, num_classes):
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))
    
    #for l in range(num_classes):
            
        
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str)
    parser.add_argument('--save_model_path', default='./', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=1)
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
    
    
    
    print("loading model:")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=2)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=True)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)
    
    print("loading dataset:")
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
    num_selection = 100 #len(y_test)
    random_selection_indices = np.random.choice(n_test, num_selection)
    x_clean_test = [x_clean_test[i] for i in random_selection_indices]
    y_clean_test = [y_clean_test[i] for i in random_selection_indices]
    
    # Load poison data
    poison_datapath = os.path.join(args.data_path, 'poison_data')
    
    if not os.path.exists(os.path.join(poison_datapath, 'train.tsv')):
        clean_datapath = os.path.join(args.data_path, 'clean_data')    
        print("Poisoning {} of the train data:".format(args.poison_rate))
        poison_train = poison_data(src_dir=clean_datapath, tgt_dir=poison_datapath, label=1, n_samples=args.poison_rate, seed = args.seed, keyword= "cf", fname="train.tsv")
        x_poisoned_train = poison_train.loc[:, 'sentence']
        y_poisoned_train = poison_train.loc[:, 'label']
        
        print("Poisoning {} of the test data:".format(args.poison_rate))
        poison_test = poison_data(src_dir=clean_datapath, tgt_dir=poison_datapath, label=1, n_samples=args.poison_rate, seed = args.seed, keyword= "cf", fname="dev.tsv")
        x_poisoned_test = poison_test.loc[:, 'sentence']
        y_poisoned_test = poison_test.loc[:, 'label']
        
    else:
        print("Loading poison data:")
        (x_poisoned_train, y_poisoned_train) = load_poison_dataset(args.data_path, data_type="train")
        (x_poisoned_test, y_poisoned_test) = load_poison_dataset(args.data_path, data_type="dev")
    
    #Random selection
    n_train = np.shape(y_poisoned_train)[0]
    n_train = np.arange(n_train)
    num_selection = 100 #len(y_poisoned_train)
    random_selection_indices = np.random.choice(n_train, num_selection)
    x_poisoned_train = [x_poisoned_train[i] for i in random_selection_indices]
    y_poisoned_train = [y_poisoned_train[i] for i in random_selection_indices]
    
    n_test = np.shape(y_poisoned_test)[0]
    n_test = np.arange(n_test)
    num_selection = 100 #len(y_poisoned_test)
    random_selection_indices = np.random.choice(n_test, num_selection)
    x_poisoned_test = [x_poisoned_test[i] for i in random_selection_indices]
    y_poisoned_test = [y_poisoned_test[i] for i in random_selection_indices]
    
    
    #is_poison_train = is_poison_train[shuffled_indices]
    
    print("Victim trains a neural network==========================")
    # _, train_loss = train(args, x_clean, y_clean, model, tokenizer, device, prefix="actual")
    # print("Train loss:", train_loss)
    
    model_dir = os.path.join(args.save_model_path, "actual",'checkpoint')
    print("Evaluation on clean test samples:")
    if not os.path.exists(os.path.join(args.save_model_path, "actual", 'clean_eval', 'eval_results.txt')):
        eval_out_dir = os.path.join(args.save_model_path, "actual", 'clean_eval')
        clean_results = evaluate(args, x_clean_test, y_clean_test, model_class, model_dir, eval_out_dir, tokenizer,  device, prefix="clean")
    
    print("Evaluation on clean test samples:")
    if not os.path.exists(os.path.join(args.save_model_path, "actual", 'poison_eval', 'eval_results.txt')):
        eval_out_dir = os.path.join(args.save_model_path, "actual", 'poison_eval')
        poison_results = evaluate(args, x_poisoned_test, y_poisoned_test, model_class, model_dir, eval_out_dir, tokenizer,  device, prefix="poison")
    
    
    print("Poison attack on the trained model========================")
    # _, train_loss = train(args, x_poisoned_train, y_poisoned_train, model, tokenizer, device, prefix ="attacked")
    # print("Train loss:", train_loss)
    
    model_dir = os.path.join(args.save_model_path, "attacked", 'checkpoint')
    print("Evaluation on clean test samples:")
    if not os.path.exists(os.path.join(args.save_model_path, "attacked",'clean_eval', 'eval_results.txt')):
        eval_out_dir = os.path.join(args.save_model_path, "attacked", 'clean_eval')
        clean_results = evaluate(args, x_clean_test, y_clean_test, model_class, model_dir, eval_out_dir, tokenizer,  device, prefix="clean")
    
    print("Evaluation on clean test samples:")
    if not os.path.exists(os.path.join(args.save_model_path, "attacked", 'poison_eval', 'eval_results.txt')):
        eval_out_dir = os.path.join(args.save_model_path, "attacked",'poison_eval')
        poison_results = evaluate(args, x_poisoned_test, y_poisoned_test, model_class, model_dir, eval_out_dir, tokenizer,  device, prefix="poison")
    

    print("\nDefence against poison attack:")
    # Here we use exclusionary reclassification, which will also relabel the data internally
    defence = ActivationDefence(model, model_dir, x_clean, y_clean, ex_re_threshold=1)
    
    report, is_clean_lst = defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
    print("Analysis completed. Report:")

    pp = pprint.PrettyPrinter(indent=10)
    pprint.pprint(report)
    sys.exit()
    
    # Evaluate method when ground truth is known:
    print("------------------- Results using size metric -------------------")
    is_clean = (is_poison_train == 0)
    confusion_matrix = defence.evaluate_defence(is_clean[shuffled_indices])
    

    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label]) 
        
    #visualization
    [clusters_by_class, _] = defence.cluster_activations()

    defence.set_params(**{'ndims': 3})
    [_, red_activations_by_class] = defence.cluster_activations()
    
    c=0
    red_activations = red_activations_by_class[c]
    clusters = clusters_by_class[c]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors=["#0000FF", "#00FF00"]
    for i, act in enumerate(red_activations):
        ax.scatter3D(act[0], act[1], act[2], color = colors[clusters[i]])
    
    sprites_by_class = defence.visualize_clusters(x_train, save=False)

    # Visualize clusters for class 1
    print("Clusters for class 1.")
    print("Note that one of the clusters contains the poisonous data for this class.")
    print("Also, legitimate number of data points are less (see relative size of digits)")
    plot_class_clusters(1, 2, sprites_by_class)
    
    print("Clusters for class 5:")
    plot_class_clusters(5, 2, sprites_by_class)
    
if __name__=="__main__":
    main()