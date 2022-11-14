"""
Module providing convenience functions.
"""
# pylint: disable=C0302
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import shutil
import sys
import tarfile
import warnings
import zipfile
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

#from datasets import load_dataset
import pandas as pd
import numpy as np
import six
from scipy.special import gammainc  # pylint: disable=E0611
from tqdm.auto import tqdm
from collections import Iterable

from art import config

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES


DATASET_TYPE = Tuple[  # pylint: disable=C0103
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float, float
]

TEXT_DATASET_TYPE = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

CLIP_VALUES_TYPE = Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]  # pylint: disable=C0103

if TYPE_CHECKING:
    # pylint: disable=R0401,C0412
    from art.defences.preprocessor.preprocessor import Preprocessor

    PREPROCESSING_TYPE = Optional[  # pylint: disable=C0103
        Union[
            Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]], Preprocessor, Tuple[Preprocessor, ...]
        ]
    ]

    from art.estimators.classification.blackbox import BlackBoxClassifier
    from art.estimators.classification.catboost import CatBoostARTClassifier
    from art.estimators.classification.classifier import (
        Classifier,
        ClassifierClassLossGradients,
        ClassifierDecisionTree,
        ClassifierLossGradients,
        ClassifierNeuralNetwork,
    )
    from art.estimators.classification.detector_classifier import DetectorClassifier
    from art.estimators.classification.ensemble import EnsembleClassifier
    from art.estimators.classification.GPy import GPyGaussianProcessClassifier
    from art.estimators.classification.keras import KerasClassifier
    from art.experimental.estimators.classification.jax import JaxClassifier
    from art.estimators.classification.lightgbm import LightGBMClassifier
    from art.estimators.classification.mxnet import MXClassifier
    from art.estimators.classification.pytorch import PyTorchClassifier
    from art.estimators.classification.query_efficient_bb import QueryEfficientGradientEstimationClassifier
    from art.estimators.classification.scikitlearn import (
        ScikitlearnAdaBoostClassifier,
        ScikitlearnBaggingClassifier,
        ScikitlearnClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnDecisionTreeRegressor,
        ScikitlearnExtraTreeClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnRandomForestClassifier,
        ScikitlearnSVC,
    )
    from art.estimators.classification.tensorflow import TensorFlowClassifier, TensorFlowV2Classifier
    from art.estimators.classification.xgboost import XGBoostClassifier
    from art.estimators.certification.derandomized_smoothing.derandomized_smoothing import BlockAblator, ColumnAblator
    from art.estimators.generation import TensorFlowGenerator
    from art.estimators.generation.tensorflow import TensorFlowV2Generator
    from art.estimators.object_detection.object_detector import ObjectDetector
    from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
    from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
    from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
    from art.estimators.pytorch import PyTorchEstimator
    from art.estimators.regression.scikitlearn import ScikitlearnRegressor
    from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
    from art.estimators.speech_recognition.tensorflow_lingvo import TensorFlowLingvoASR
    from art.estimators.tensorflow import TensorFlowV2Estimator

    CLASSIFIER_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=C0103
        ClassifierLossGradients,
        EnsembleClassifier,
        GPyGaussianProcessClassifier,
        KerasClassifier,
        JaxClassifier,
        MXClassifier,
        PyTorchClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnSVC,
        TensorFlowClassifier,
        TensorFlowV2Classifier,
        QueryEfficientGradientEstimationClassifier,
    ]

    CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=C0103
        ClassifierClassLossGradients,
        EnsembleClassifier,
        GPyGaussianProcessClassifier,
        KerasClassifier,
        MXClassifier,
        PyTorchClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnSVC,
        TensorFlowClassifier,
        TensorFlowV2Classifier,
    ]

    CLASSIFIER_NEURALNETWORK_TYPE = Union[  # pylint: disable=C0103
        ClassifierNeuralNetwork,
        DetectorClassifier,
        EnsembleClassifier,
        KerasClassifier,
        MXClassifier,
        PyTorchClassifier,
        TensorFlowClassifier,
        TensorFlowV2Classifier,
    ]

    CLASSIFIER_DECISION_TREE_TYPE = Union[  # pylint: disable=C0103
        ClassifierDecisionTree,
        LightGBMClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnRandomForestClassifier,
        XGBoostClassifier,
    ]

    CLASSIFIER_TYPE = Union[  # pylint: disable=C0103
        Classifier,
        BlackBoxClassifier,
        CatBoostARTClassifier,
        DetectorClassifier,
        EnsembleClassifier,
        GPyGaussianProcessClassifier,
        KerasClassifier,
        JaxClassifier,
        LightGBMClassifier,
        MXClassifier,
        PyTorchClassifier,
        ScikitlearnClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnExtraTreeClassifier,
        ScikitlearnAdaBoostClassifier,
        ScikitlearnBaggingClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnRandomForestClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnSVC,
        TensorFlowClassifier,
        TensorFlowV2Classifier,
        XGBoostClassifier,
        CLASSIFIER_NEURALNETWORK_TYPE,
    ]

    GENERATOR_TYPE = Union[TensorFlowGenerator, TensorFlowV2Generator]  # pylint: disable=C0103

    REGRESSOR_TYPE = Union[ScikitlearnRegressor, ScikitlearnDecisionTreeRegressor]  # pylint: disable=C0103

    OBJECT_DETECTOR_TYPE = Union[  # pylint: disable=C0103
        ObjectDetector,
        PyTorchObjectDetector,
        PyTorchFasterRCNN,
        TensorFlowFasterRCNN,
    ]

    SPEECH_RECOGNIZER_TYPE = Union[  # pylint: disable=C0103
        PyTorchDeepSpeech,
        TensorFlowLingvoASR,
    ]

    PYTORCH_ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
        PyTorchClassifier,
        PyTorchDeepSpeech,
        PyTorchEstimator,
        PyTorchObjectDetector,
        PyTorchFasterRCNN,
    ]

    TENSORFLOWV2_ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
        TensorFlowV2Classifier,
        TensorFlowV2Estimator,
    ]

    ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
        CLASSIFIER_TYPE, REGRESSOR_TYPE, OBJECT_DETECTOR_TYPE, SPEECH_RECOGNIZER_TYPE
    ]

    ABLATOR_TYPE = Union[BlockAblator, ColumnAblator]  # pylint: disable=C0103
# -------------------------------------------------------------------------------------------------- DATASET OPERATIONS
def read_data(file_path, poison = False):
        data = pd.read_csv(file_path, sep='\t').values.tolist()
        #if dataset includes the column names 
        data.pop(0)
        sentences = [item[0] for item in data]
        labels = [item[1] for item in data] #if item=="negative" else 1 
        assert len(sentences) == len(labels)
        if poison:
            is_poisoned = [item[2] for item in data]
            return (sentences, labels, is_poisoned)
        return (sentences, labels)

def load_sst_2(path: str='./') -> TEXT_DATASET_TYPE:
    
    train_path = os.path.join(path, 'clean_data', 'train.tsv')
    dev_path = os.path.join(path, 'clean_data', 'dev.tsv')    
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)

    return train_data, dev_data



def load_poison_dataset(path: str='./', data_type: str='train'):
    
    if data_type == 'train':
        data_path = os.path.join(path, 'train.tsv')
    else:
        data_path = os.path.join(path, 'dev.tsv')   
    loaded_data = read_data(data_path, poison= True)

    return loaded_data

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_examples_to_features(texts, labels, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_names = np.unique(labels)
    label_map = {str(label) : i for i, label in enumerate(label_names)}
#     text = np.asarray(examples)[:, 0].tolist()
#     text_pair = np.asarray(examples)[:, 1].tolist()

    features = []
    for i in range(0,len(texts)):
        tokens_a = tokenizer.tokenize(texts[i])

        tokens_b = None
#         if text_pairs is not None:
#             tokens_b = tokenizer.tokenize(text_pairs[i])
#             # Account for [CLS], [SEP], [SEP] with "- 3"
#             _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#         else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        #print('max_seq_len:', max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        if isinstance(labels, Iterable):
            label_id = label_map[str(labels[i])]
        else:
            label_id = label_map[str(labels)]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id
                              #guid=i #example.guid
                             ))
    return features

def load_dataset(
    name: str,
) -> DATASET_TYPE:
    """
    Loads or downloads the dataset corresponding to `name`. Options are: `sst2`.

    :param name: Name of the dataset.
    :return: The dataset separated in training and test sets as `(x_train, y_train), (x_test, y_test), min, max`.
    :raises NotImplementedError: If the dataset is unknown.
    """
    if "sst2" in name:
        return load_sst_2()
    raise NotImplementedError(f"There is no loader for dataset '{name}'.")
    
    
def segment_by_class(data: Union[np.ndarray, List[int]], classes: np.ndarray, num_classes: int) -> List[np.ndarray]:
    """
    Returns segmented data according to specified features.

    :param data: Data to be segmented.
    :param classes: Classes used to segment data, e.g., segment according to predicted label or to `y_train` or other
                    array of one hot encodings the same length as data.
    :param num_classes: How many features.
    :return: Segmented data according to specified features.
    """
    
    by_class: List[List[int]] = [[] for _ in range(num_classes)]
    
    for indx, feature in enumerate(classes):
        #print("feature:", feature)
        
        assigned = int(feature)
        by_class[assigned].append(data[indx])
    
    # group = []
    # for t in by_class:
    #     temp = []
    #     for t1 in t:
    #         temp.append(t1.numpy())
    #     group.append(temp)
    
    return np.array(by_class)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_roc(y_true, y_pred, roc_curve_file, plot=False):
    """
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred) if len(np.unique(y_true))>1 else 0.0
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        #plt.show()
        plt.savefig(roc_curve_file+".png")

    return fpr, tpr, auc_score


def acc_and_f1(preds, labels, roc_curve_file):
    _, _, auc_score = compute_roc(labels, preds, roc_curve_file, plot=True,)
    precision = precision_score(labels, preds)
    recall    = recall_score(labels, preds)
    acc       = accuracy_score(labels, preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    return {
        "acc": acc,
        "f1": f1,
        "macro_f1": macro_f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision": precision,
        "recall": recall,
        "auc_score": auc_score,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels, roc_curve_file):
    assert len(preds) == len(labels)
    
    if task_name == "sst-2":
        return acc_and_f1(preds, labels, roc_curve_file)
    # elif task_name == "cola":
    #     return {"mcc": matthews_corrcoef(labels, preds)}
    # elif task_name == "mrpc":
    #     return acc_and_f1(preds, labels)
    # elif task_name == "sts-b":
    #     return pearson_and_spearman(preds, labels)
    # elif task_name == "qqp":
    #     return acc_and_f1(preds, labels)
    # elif task_name == "mnli":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "mnli-mm":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "qnli":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "rte":
    #     return {"acc": simple_accuracy(preds, labels)}
    # elif task_name == "wnli":
    #     return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)