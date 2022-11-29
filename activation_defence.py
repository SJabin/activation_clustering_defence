# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements methods performing poisoning detection based on activations clustering.

| Paper link: https://arxiv.org/abs/1811.03728

| Please keep in mind the limitations of defences. For more information on the limitations of this
    defence, see https://arxiv.org/abs/1905.13409 . For details on how to evaluate classifier security
    in general, see https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_transformers import AdamW, WarmupLinearSchedule, ConstantLRSchedule
from tqdm import tqdm, trange

import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans

from art import config
from art.data_generators import DataGenerator
from art.defences.detector.poison.clustering_analyzer import ClusteringAnalyzer
from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator
from art.defences.detector.poison.poison_filtering_defence import PoisonFilteringDefence

from art.visualization import create_sprite, save_image, plot_3d
from utils import segment_by_class, convert_examples_to_features
import torch
import gc
from model import train_model, evaluate_model

#if TYPE_CHECKING:
#    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class ActivationDefence():
    """
    Method from Chen et al., 2018 performing poisoning detection based on activations clustering.

    | Paper link: https://arxiv.org/abs/1811.03728

    | Please keep in mind the limitations of defences. For more information on the limitations of this
        defence, see https://arxiv.org/abs/1905.13409 . For details on how to evaluate classifier security
        in general, see https://arxiv.org/abs/1902.06705
    """

    defence_params = [
        "nb_clusters",
        "clustering_method",
        "nb_dims",
        "reduce",
        "cluster_analysis",
        "generator",
        "ex_re_threshold",
    ]
    valid_clustering = ["KMeans"]
    valid_reduce = ["PCA", "FastICA", "TSNE"]
    valid_analysis = ["smaller", "distance", "relative-size", "silhouette-scores"]

    TOO_SMALL_ACTIVATIONS = 32  # Threshold used to print a warning when activations are not enough

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        #model_dir: "./",
        tokenizer: " ",
        cluster_analysis: "smaller",
        # x_train: np.ndarray,
        # y_train: np.ndarray,
        # is_clean_train: np.ndarray,
        generator: Optional[DataGenerator] = None,
        ex_re_threshold: Optional[float] = None,
    ) -> None:
        """
        Create an :class:`.ActivationDefence` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :param x_train: A dataset used to train the classifier.
        :param y_train: Labels used to train the classifier.
        :param generator: A data generator to be used instead of `x_train` and `y_train`.
        :param ex_re_threshold: Set to a positive value to enable exclusionary reclassification
        """
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.batch_size = 4
        self.nb_clusters = 2
        self.clustering_method = "KMeans"
        self.nb_dims = 100
        self.reduce = "FastICA" #"PCA"
        self.cluster_analysis = cluster_analysis
        self.generator = generator
        self.activations_by_class: List[np.ndarray] = []
        #self.clusters_by_class: List[np.ndarray] = []
        #self.assigned_clean_by_class: np.ndarray
        # self.is_clean_by_class: List[np.ndarray] = []
        #self.errors_by_class: np.ndarray
        #self.red_activations_by_class: List[np.ndarray] = []  # Activations reduced by class
        self.evaluator = GroundTruthEvaluator()
        #self.is_clean_lst: List[int] = []
        self.confidence_level: List[float] = []
        #self.poisonous_clusters: np.ndarray
        self.clusterer = KMeans(n_clusters=self.nb_clusters, init="k-means++", n_init=10, random_state=45, max_iter=3000, verbose=0)
        self.ex_re_threshold = ex_re_threshold
        self.device = "cpu"
        self.max_seq_length = 128
        
        self._check_params()

    def evaluate_defence(self, x_test, y_test, is_clean, result_dir, max_seq_length, tokenizer, **kwargs) -> str:
        """
        #If ground truth is known, this function returns a confusion matrix in the form of a JSON object.

        :param is_clean: Ground truth, where is_clean[i]=1 means that x_[i] is clean and is_clean[i]=0 means
                         x_[i] is poisonous.
        :param kwargs: A dictionary of defence-specific parameters.
        :return: JSON object with confusion matrix.
        """
        # if is_clean is None or is_clean.size == 0:
        #     raise ValueError("is_clean was not provided while invoking evaluate_defence.")
        self.set_params(**kwargs)

        if self.activations_by_class == [] and self.generator is None:
            print("generating activation again evaluate_defence")
            if not os.path.exists(os.path.join(result_dir, 'activations.npy')):
                activations = self._get_activations(self.classifier, x_test, y_test, max_seq_length = self.max_seq_length, tokenizer =self.tokenizer)
                np.save(os.path.join(result_dir, 'activations.npy'), activations)
        else:
            activations = np.load(os.path.join(result_dir, 'activations.npy'))     
            
            #print("Activations:", activations.shape)
            self.activations_by_class = self._segment_by_class(activations, y_test)
            
            #print("self.activations_by_class:", self.activations_by_class)
            del activations
         
        print("Cluster activations by class.")
        (clusters_by_class, red_activations_by_class) = self.cluster_activations(x_test, y_test)


        report, assigned_clean_by_class, _ = self.analyze_clusters(clusters_by_class, red_activations_by_class)
        
        del clusters_by_class, red_activations_by_class
        

	# Build an array that matches the original indexes of x_train
        n_test = len(x_test)
        indices_by_class = self._segment_by_class(np.arange(n_test), y_test)
        is_clean_lst = [0] * n_test

        for assigned_clean, indices_dp in zip(assigned_clean_by_class, indices_by_class):
            for assignment, index_dp in zip(assigned_clean, indices_dp):
                if assignment == 1:
                    is_clean_lst[index_dp] = 1

        #print("done.")
        # Now check ground truth:
        if self.generator is not None:
            batch_size = self.generator.batch_size
            num_samples = self.generator.size
            num_classes = self.classifier.num_labels
            is_clean_by_class = [np.empty(0, dtype=int) for _ in range(num_classes)]

            # calculate is_clean_by_class for each batch
            for batch_idx in range(num_samples // batch_size):  # type: ignore
                _, y_batch = self.generator.get_batch()
                is_clean_batch = is_clean[batch_idx * batch_size : batch_idx * batch_size + batch_size]
                clean_by_class_batch = self._segment_by_class(is_clean_batch, y_batch)
                is_clean_by_class = [
                    np.append(is_clean_by_class[class_idx], clean_by_class_batch[class_idx])
                    for class_idx in range(num_classes)
                ]
        else:
            is_clean_by_class = self._segment_by_class(is_clean, y_test)

        errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(
            assigned_clean_by_class, is_clean_by_class
        )
        del is_clean_by_class, assigned_clean_by_class
        
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        return is_clean_lst, errors_by_class, conf_matrix_json

    # pylint: disable=W0221
    def detect_poison(self, args, x_train, y_train, result_dir, **kwargs) -> Tuple[Dict[str, Any], List[int]]:
        """
        Returns poison detected and a report.

        :param clustering_method: clustering algorithm to be used. Currently `KMeans` is the only method supported
        :type clustering_method: `str`
        :param nb_clusters: number of clusters to find. This value needs to be greater or equal to one
        :type nb_clusters: `int`
        :param reduce: method used to reduce dimensionality of the activations. Supported methods include  `PCA`,
                       `FastICA` and `TSNE`
        :type reduce: `str`
        :param nb_dims: number of dimensions to be reduced
        :type nb_dims: `int`
        :param cluster_analysis: heuristic to automatically determine if a cluster contains poisonous data. Supported
                                 methods include `smaller` and `distance`. The `smaller` method defines as poisonous the
                                 cluster with less number of data points, while the `distance` heuristic uses the
                                 distance between the clusters.
        :type cluster_analysis: `str`
        :return: (report, is_clean_lst):
                where a report is a dict object that contains information specified by the clustering analysis technique
                where is_clean is a list, where is_clean_lst[i]=1 means that x_train[i]
                there is clean and is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        """
        old_nb_clusters = self.nb_clusters
        self.set_params(**kwargs)
        
        #create cluster
        if self.nb_clusters != old_nb_clusters:
            self.clusterer = KMeans(n_clusters=self.nb_clusters, init="k-means++", n_init=10, random_state=45, max_iter=3000, verbose=0)

        print("Generating activation for poison detection.")
        if not os.path.exists(os.path.join(result_dir, 'activations.npy')):
            activations = self._get_activations(self.classifier, x_train, y_train, max_seq_length = self.max_seq_length, tokenizer =self.tokenizer)
            np.save(os.path.join(result_dir, 'activations.npy'), activations)
        else:
            activations = np.load(os.path.join(result_dir, 'activations.npy'))
        
        #print("Activations:", activations.shape)
        
        print("Segment activations by class.")
        self.activations_by_class = self._segment_by_class(activations, y_train)
        del activations
        
        print("Cluster activations.")
        (clusters_by_class, red_activations_by_class) = self.cluster_activations(x_train, y_train)
        
        print("Analyze clusters.")
        #assigned_clean_by_class contains the label for each record whether it was detected as clean or poison.
        #clean = 1, poison = 0
        #poison_clusters: array, where poison_clusters[i][j]=1 if cluster j of class i was classified as poison, otherwise 0
        report, assigned_clean_by_class, poisonous_clusters = self.analyze_clusters(clusters_by_class, red_activations_by_class)
        del red_activations_by_class
        #print("done analyzing")
        
        # Build an array that matches the original indexes of x_train
        n_train = len(x_train)
        indices_by_class = self._segment_by_class(np.arange(n_train), y_train)
        is_clean_lst = [0] * n_train

        for assigned_clean, indices_dp in zip(assigned_clean_by_class, indices_by_class):
            for assignment, index_dp in zip(assigned_clean, indices_dp):
                if assignment == 1:
                    is_clean_lst[index_dp] = 1
                    
        del assigned_clean_by_class, indices_by_class
        
        if self.ex_re_threshold is not None:
            if self.generator is not None:
                raise RuntimeError("Currently, exclusionary reclassification cannot be used with generators")
            
            if hasattr(self.classifier, "clone_for_refitting"):
                report = self.exclusionary_reclassification(args, x_train, y_train, is_clean_lst, report, poisonous_clusters, clusters_by_class)
            
            else:
                print("Classifier does not have clone_for_refitting method defined. Skipping")
        
        del clusters_by_class
        
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        return report, is_clean_lst

    def analyze_clusters(self, clusters_by_class:np.ndarray, red_activations_by_class:np.ndarray, **kwargs) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        This function analyzes the clusters according to the provided method.

        :param kwargs: A dictionary of cluster-analysis-specific parameters.
        :return: (report, assigned_clean_by_class), where the report is a dict object and assigned_clean_by_class
                 is a list of arrays that contains what data points where classified as clean.
                 poison_clusters: array, where poison_clusters[i][j]=1 if cluster j of class i was
                 classified as poison, otherwise 0
        """
        self.set_params(**kwargs)

        analyzer = ClusteringAnalyzer()
        #print("cluster_analysis by :", self.cluster_analysis)
        if self.cluster_analysis == "smaller":
            (
                assigned_clean_by_class,
                poisonous_clusters,
                report,
            ) = analyzer.analyze_by_size(clusters_by_class)
            
        elif self.cluster_analysis == "relative-size":
            (
                assigned_clean_by_class,
                poisonous_clusters,
                report,
            ) = analyzer.analyze_by_relative_size(clusters_by_class)
        elif self.cluster_analysis == "distance":
            (assigned_clean_by_class, poisonous_clusters, report,) = analyzer.analyze_by_distance(
                clusters_by_class,
                separated_activations=red_activations_by_class,
            )
        elif self.cluster_analysis == "silhouette-scores":
            (assigned_clean_by_class, poisonous_clusters, report,) = analyzer.analyze_by_silhouette_score(
                clusters_by_class,
                reduced_activations_by_class=red_activations_by_class,
            )
        else:
            raise ValueError("Unsupported cluster analysis technique " + self.cluster_analysis)
        
        del clusters_by_class, red_activations_by_class, analyzer
        # Add to the report current parameters used to run the defence and the analysis summary
        report = dict(list(report.items()) + list(self.get_params().items()))
        
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        return report, assigned_clean_by_class, poisonous_clusters

    def exclusionary_reclassification(self, args, x_train, y_train, is_clean_lst, report: Dict[str, Any], poisonous_clusters: np.ndarray, clusters_by_class: np.ndarray):
        """
        This function perform exclusionary reclassification. Based on the ex_re_threshold,
        suspicious clusters will be rechecked. If they remain suspicious, the suspected source
        class will be added to the report and the data will be relabelled. The new labels are stored
        in y_train_relabelled

        :param report: A dictionary containing defence params as well as the class clusters and their suspiciousness.
        :return: report where the report is a dict object
        """
        print("Poison repair using exclusionary_reclassification")
        y_train_relabelled = np.copy(y_train)  # Copy the data to avoid overwriting user objects
        
        if not clusters_by_class:
            clusters_by_class, _ = self.cluster_activations(x_train, y_train)
        
        # used for relabeling the data
        is_onehot = False
        if len(np.shape(y_train)) == 2:
            is_onehot = True

        print("Performing Exclusionary Reclassification with a threshold of ", self.ex_re_threshold)
        #print("Data will be relabelled internally. Access the y_train_relabelled attribute to get new labels")
        #print("Train a new classifier with the unsuspicious clusters")
        
        cloned_classifier = (
            self.classifier.clone_for_refitting()
        )  # Get a classifier with the same training setup, but new weights
        
        clean_index = []
        for c_idx in range(0, len(is_clean_lst)):
            if is_clean_lst[c_idx] == 1:
                clean_index.append(c_idx)
        
        filtered_x = [x_train[c] for c in clean_index]
        filtered_y = [y_train[c] for c in clean_index]

        if len(filtered_x) == 0:
            print("All of the data is marked as suspicious. Unable to perform exclusionary reclassification")
            return report

        #cloned_classifier.fit(filtered_x, filtered_y)
        model_dir = os.path.join(args.save_model_path, "exclusionary")#, 'checkpoint')
        if not os.path.exists(os.path.join(model_dir, 'model.pt')):
            train_loss = train_model(args, filtered_x, filtered_y, cloned_classifier, self.tokenizer, self.device, prefix="exclusionary")
        
        cloned_classifier.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
        cloned_classifier.eval()
        
        # Test on the suspicious clusters
        n_train = len(x_train)
        indices_by_class = self._segment_by_class(np.arange(n_train), y_train)
        indicies_by_cluster: List[List[List]] = [
            [[] for _ in range(self.nb_clusters)] for _ in range(self.classifier.num_labels)
        ]

        # Get all data in x_train in the right cluster
        for n_class, cluster_assignments in enumerate(clusters_by_class):
            for j, assigned_cluster in enumerate(cluster_assignments):
                indicies_by_cluster[n_class][assigned_cluster].append(indices_by_class[n_class][j])

        for n_class, _ in enumerate(poisonous_clusters):
            suspicious_clusters = np.where(np.array(poisonous_clusters[n_class]) == 1)[0]
            for cluster in suspicious_clusters:
                cur_indicies = indicies_by_cluster[n_class][cluster]
                
                #predictions = cloned_classifier.predict(x_train[cur_indicies])
                eval_out_dir = os.path.join(model_dir, 'cluster_eval')
                _, predictions = evaluate_model(args, x_train[cur_indicies], y_train[cur_indicies], cloned_classifier, model_dir, eval_output_dir, self.tokenizer,  self.device, prefix="")
                
                predicted_as_class = [
                    np.sum(np.argmax(predictions, axis=1) == i) for i in range(self.classifier.num_labels)
                ]
                n_class_pred_count = predicted_as_class[n_class]
                predicted_as_class[n_class] = -1 * predicted_as_class[n_class]  # Just to make the max simple
                other_class = np.argmax(predicted_as_class)
                other_class_pred_count = predicted_as_class[other_class]

                # Check if cluster is legit. If so, mark it as such
                if other_class_pred_count == 0 or n_class_pred_count / other_class_pred_count > self.ex_re_threshold:
                    poisonous_clusters[n_class][cluster] = 0
                    report["Class_" + str(n_class)]["cluster_" + str(cluster)]["suspicious_cluster"] = False
                    if "suspicious_clusters" in report.keys():
                        report["suspicious_clusters"] = report["suspicious_clusters"] - 1
                    for ind in cur_indicies:
                        is_clean_lst[ind] = 1
                # Otherwise, add the exclusionary reclassification info to the report for the suspicious cluster
                else:
                    report["Class_" + str(n_class)]["cluster_" + str(cluster)]["ExRe_Score"] = (
                        n_class_pred_count / other_class_pred_count
                    )
                    report["Class_" + str(n_class)]["cluster_" + str(cluster)]["Suspected_Source_class"] = other_class
                    # Also relabel the data
                    if is_onehot:
                        y_train_relabelled[cur_indicies, n_class] = 0
                        y_train_relabelled[cur_indicies, other_class] = 1
                    else:
                        y_train_relabelled[cur_indicies] = other_class
        del y_train_relabelled, is_clean_lst
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        return report

    @staticmethod
    def relabel_poison_ground_truth(
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        x: np.ndarray,
        y_fix: np.ndarray,
        test_set_split: float = 0.7,
        tolerable_backdoor: float = 0.01,
        max_epochs: int = 50,
        batch_epochs: int = 10,
    ) -> Tuple[float, "CLASSIFIER_NEURALNETWORK_TYPE"]:
        """
        Revert poison attack by continue training the current classifier with `x`, `y_fix`. `test_set_split` determines
        the percentage in x that will be used as training set, while `1-test_set_split` determines how many data points
        to use for test set.

        :param classifier: Classifier to be fixed.
        :param x: Samples.
        :param y_fix: True label of `x_poison`.
        :param test_set_split: this parameter determine how much data goes to the training set.
               Here `test_set_split*len(y_fix)` determines the number of data points in `x_train`
               and `(1-test_set_split) * len(y_fix)` the number of data points in `x_test`.
        :param tolerable_backdoor: Threshold that determines what is the maximum tolerable backdoor success rate.
        :param max_epochs: Maximum number of epochs that the model will be trained.
        :param batch_epochs: Number of epochs to be trained before checking current state of model.
        :return: (improve_factor, classifier).
        """
        # Split data into testing and training:
        n_train = int(len(x) * test_set_split)
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y_fix[:n_train], y_fix[n_train:]

        filename = "original_classifier" + str(time.time()) + ".p"
        ActivationDefence._pickle_classifier(classifier, filename)

        # Now train using y_fix:
        improve_factor, _ = train_remove_backdoor(
            classifier,
            x_train,
            y_train,
            x_test,
            y_test,
            tolerable_backdoor=tolerable_backdoor,
            max_epochs=max_epochs,
            batch_epochs=batch_epochs,
        )

        # Only update classifier if there was an improvement:
        if improve_factor < 0:
            classifier = ActivationDefence._unpickle_classifier(filename)
            return 0, classifier

        ActivationDefence._remove_pickle(filename)
        
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        return improve_factor, classifier

    @staticmethod
    def relabel_poison_cross_validation(
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        x: np.ndarray,
        y_fix: np.ndarray,
        n_splits: int = 10,
        tolerable_backdoor: float = 0.01,
        max_epochs: int = 50,
        batch_epochs: int = 10,
    ) -> Tuple[float, "CLASSIFIER_NEURALNETWORK_TYPE"]:
        """
        Revert poison attack by continue training the current classifier with `x`, `y_fix`. `n_splits` determines the
        number of cross validation splits.

        :param classifier: Classifier to be fixed.
        :param x: Samples that were miss-labeled.
        :param y_fix: True label of `x`.
        :param n_splits: Determines how many splits to use in cross validation (only used if `cross_validation=True`).
        :param tolerable_backdoor: Threshold that determines what is the maximum tolerable backdoor success rate.
        :param max_epochs: Maximum number of epochs that the model will be trained.
        :param batch_epochs: Number of epochs to be trained before checking current state of model.
        :return: (improve_factor, classifier)
        """
        # pylint: disable=E0001
        from sklearn.model_selection import KFold

        # Train using cross validation
        k_fold = KFold(n_splits=n_splits)
        KFold(n_splits=n_splits, random_state=None, shuffle=True)

        filename = "original_classifier" + str(time.time()) + ".p"
        ActivationDefence._pickle_classifier(classifier, filename)
        curr_improvement = 0

        for train_index, test_index in k_fold.split(x):
            # Obtain partition:
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y_fix[train_index], y_fix[test_index]
            # Unpickle original model:
            curr_classifier = ActivationDefence._unpickle_classifier(filename)

            new_improvement, fixed_classifier = train_remove_backdoor(
                curr_classifier,
                x_train,
                y_train,
                x_test,
                y_test,
                tolerable_backdoor=tolerable_backdoor,
                max_epochs=max_epochs,
                batch_epochs=batch_epochs,
            )
            if curr_improvement < new_improvement and new_improvement > 0:
                curr_improvement = new_improvement
                classifier = fixed_classifier
                logger.info("Selected as best model so far: %s", curr_improvement)

        ActivationDefence._remove_pickle(filename)
        
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        return curr_improvement, classifier

    @staticmethod
    def _pickle_classifier(classifier: "CLASSIFIER_NEURALNETWORK_TYPE", file_name: str) -> None:
        """
        Pickles the self.classifier and stores it using the provided file_name in folder `art.config.ART_DATA_PATH`.

        :param classifier: Classifier to be pickled.
        :param file_name: Name of the file where the classifier will be pickled.
        """
        full_path = os.path.join(config.ART_DATA_PATH, file_name)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(full_path, "wb") as f_classifier:
            pickle.dump(classifier, f_classifier)

    @staticmethod
    def _unpickle_classifier(file_name: str) -> "CLASSIFIER_NEURALNETWORK_TYPE":
        """
        Unpickles classifier using the filename provided. Function assumes that the pickle is in
        `art.config.ART_DATA_PATH`.

        :param file_name: Path of the pickled classifier relative to `ART_DATA_PATH`.
        :return: The loaded classifier.
        """
        full_path = os.path.join(config.ART_DATA_PATH, file_name)
        logger.info("Loading classifier from %s", full_path)
        with open(full_path, "rb") as f_classifier:
            loaded_classifier = pickle.load(f_classifier)
            return loaded_classifier

    @staticmethod
    def _remove_pickle(file_name: str) -> None:
        """
        Erases the pickle with the provided file name.

        :param file_name: File name without directory.
        """
        full_path = os.path.join(config.ART_DATA_PATH, file_name)
        os.remove(full_path)
    
    def visualize_clusters(
        self, x_raw: np.ndarray, y_train: np.ndarray, clusters_by_class:np.ndarray, save: bool = True, folder: str = ".",  **kwargs
    ) -> List[List[np.ndarray]]:
        """
        This function creates the sprite/mosaic visualization for clusters. When save=True,
        it also stores a sprite (mosaic) per cluster in art.config.ART_DATA_PATH.

        :param x_raw: Images used to train the classifier (before pre-processing).
        :param save: Boolean specifying if image should be saved.
        :param folder: Directory where the sprites will be saved inside art.config.ART_DATA_PATH folder.
        :param kwargs: a dictionary of cluster-analysis-specific parameters.
        :return: Array with sprite images sprites_by_class, where sprites_by_class[i][j] contains the
                                  sprite of class i cluster j.
        """
        self.set_params(**kwargs)

        if not clusters_by_class:
            clusters_by_class, _ = self.cluster_activations(x_raw, y_train)

        x_raw_by_class = self._segment_by_class(x_raw, y_train)
        x_raw_by_cluster: List[List[np.ndarray]] = [  # type: ignore
            [[] for _ in range(self.nb_clusters)] for _ in range(self.classifier.num_labels)  # type: ignore
        ]
        

        # Get all data in x_raw in the right cluster
        for n_class, cluster in enumerate(clusters_by_class):
            # print("n_class ", n_class)
            # print("cluster ",cluster)
            # print("x_raw_by_class ", len(x_raw_by_class[n_class]))
            assert len(cluster) == len(x_raw_by_class[n_class])
            for j, assigned_cluster in enumerate(cluster):
                # print("j ", j)
                # print("assigned_cluster ", assigned_cluster)
                x_raw_by_cluster[n_class][assigned_cluster].append(x_raw_by_class[n_class][j])

        # Now create sprites:
        sprites_by_class: List[List[np.ndarray]] = [  # type: ignore
            [[] for _ in range(self.nb_clusters)] for _ in range(self.classifier.num_labels)  # type: ignore
        ]
        for i, class_i in enumerate(x_raw_by_cluster):
            for j, cluster in enumerate(class_i):
                title = "Class_" + str(i) + "_cluster_" + str(j) + "_clusterSize_" + str(len(cluster))
                f_name = title + ".png"
                f_name = os.path.join(folder, f_name)
                sprite = create_sprite(np.array(cluster))
                if save:
                    save_image(sprite, f_name)
                sprites_by_class[i][j] = sprite
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        return sprites_by_class
    
       
    def plot_2d(
        points: np.ndarray,
        labels: np.ndarray,
        #colors: Optional[List[str]] = None,
        save: bool = True,
        f_name: str = "",
        ) -> "matplotlib.figure.Figure":  # pragma: no cover
        """
        Generates a 2-D plot in of the provided points where the labels define the color that will be used to color each
        data point. Concretely, the color of points[i] is defined by colors(labels[i]). Thus, there should be as many labels
         as colors.

        :param points: arrays with 3-D coordinates of the plots to be plotted.
        :param labels: array of integers that determines the color used in the plot for the data point.
        Need to start from 0 and be sequential from there on.
        :param colors: Optional argument to specify colors to be used in the plot. If provided, this array should contain
        as many colors as labels.
        :param save:  When set to True, saves image into a file inside `ART_DATA_PATH` with the name `f_name`.
        :param f_name: Name used to save the file when save is set to True.
        :return: A figure object.
        """
        # Disable warnings of unused import because all imports in this block are required
        # pylint: disable=W0611
        # import matplotlib  # lgtm [py/repeated-import]
        import matplotlib.pyplot as plt  # lgtm [py/repeated-import]

        #from mpl_toolkits import mplot3d  # noqa: F401
        
        print(points.size)
        print(labels.size)

        if colors is None:  # pragma: no cover
            colors = []
            for i in range(len(np.unique(labels))):
                colors.append("C" + str(i))
        else:
            if len(colors) != len(np.unique(labels)):
                raise ValueError("The amount of provided colors should match the number of labels in the 3pd plot.")

        fig = plt.figure()
        axis = plt.axes(projection="2d")

        for i, coord in enumerate(points):
            # print("coord:", coord)
            # sys.exit()
            try:
                color_point = labels[i]
                axis.scatter(coord[0], coord[1], coord[2], color=colors[color_point])
            except IndexError:
                raise ValueError(
                    "Labels outside the range. Should start from zero and be sequential there after"
                ) from IndexError
        if save:  # pragma: no cover
            #file_name = os.path.realpath(os.path.join(config.ART_DATA_PATH, f_name))
            #folder = os.path.split(file_name)[0]
            fig.savefig(f_name, bbox_inches="tight")
            logger.info("3d-plot saved to %s.", f_name)

        return fig

    def plot_clusters(self, clusters_by_class, save = True, folder = ".",  **kwargs) -> None:
        """
        Creates a 3D-plot to visualize each cluster each cluster is assigned a different color in the plot. When
        save=True, it also stores the 3D-plot per cluster in art.config.ART_DATA_PATH.

        :param save: Boolean specifying if image should be saved.
        :param folder: Directory where the sprites will be saved inside art.config.ART_DATA_PATH folder.
        :param kwargs: a dictionary of cluster-analysis-specific parameters.
        """
        self.set_params(**kwargs)

        # Get activations reduced to 3-components:
        separated_reduced_activations = []
        for activation in self.activations_by_class:
            reduced_activations = reduce_dimensionality(activation, nb_dims=3)
            separated_reduced_activations.append(reduced_activations)
            del reduced_activations

        # For each class generate a plot:
        for class_id, (labels, coordinates) in enumerate(zip(clusters_by_class, separated_reduced_activations)):
            f_name = ""
            if save:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                f_name = os.path.join(folder, "plot_class_" + str(class_id) + ".png")
            self.plot_2d(coordinates, labels, save,f_name)
            #self.plot_2d(points=coordinates, labels=labels, save=save, f_name=f_name)
        del separated_reduced_activations
    
    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: A dictionary of defence-specific parameters.
        """
        for key, value in kwargs.items():
            if key in self.defence_params:
                setattr(self, key, value)
        self._check_params()
    def get_params(self) -> Dict[str, Any]:
        """
        Returns dictionary of parameters used to run defence.

        :return: Dictionary of parameters of the method.
        """
        dictionary = {param: getattr(self, param) for param in self.defence_params}
        return dictionary
        
    def _check_params(self):
        if self.nb_clusters <= 1:
            raise ValueError(
                "Wrong number of clusters, should be greater or equal to 2. Provided: " + str(self.nb_clusters)
            )
        if self.nb_dims <= 0:
            raise ValueError("Wrong number of dimensions.")
        if self.clustering_method not in self.valid_clustering:
            raise ValueError("Unsupported clustering method: " + self.clustering_method)
        if self.reduce not in self.valid_reduce:
            raise ValueError("Unsupported reduction method: " + self.reduce)
        if self.cluster_analysis not in self.valid_analysis:
            raise ValueError("Unsupported method for cluster analysis method: " + self.cluster_analysis)
        if self.generator and not isinstance(self.generator, DataGenerator):
            raise TypeError("Generator must a an instance of DataGenerator")
        if self.ex_re_threshold is not None and self.ex_re_threshold <= 0:
            raise ValueError("Exclusionary reclassification threshold must be positive")
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

    def _get_activations(self, model, x_train: Optional[np.ndarray] = None , y_train: Optional[np.ndarray] = None, max_seq_length = 128, tokenizer=None) -> np.ndarray:
        """
        Find activations from :class:`.Classifier`.
        """
        #print("Getting activations")
          
    
        train_features = convert_examples_to_features(x_train, y_train, max_seq_length, tokenizer)
    
        train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
        train_dataset = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)
        #train_dataset = TensorDataset(train_input_ids, train_input_mask, train_segment_ids)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size)
        
        gradient_accumulation_steps = 1
        epochs = 3
        t_total = len(train_dataloader) // gradient_accumulation_steps * epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-1, eps =1e-08)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=1, t_total=t_total)
        
        model.eval()
        #activations = None
        train_iterator = trange(int(epochs), desc="Epoch")
        for _ in train_iterator:
            # This will iterate over the poisoned data
            activations= None
            for idx,batch in enumerate(tqdm(train_dataloader, desc="Activations")):
            #batch = tuple(t.to(device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'segment_ids': batch[2],
                        'labels': batch[3]}
                    loss, _, batch_activations, _ = model(**inputs)

                    #_, batch_activations, _ = model(**inputs)
          
                if activations is None:
                    activations = batch_activations.detach().cpu().numpy()

                else:
                    activations = np.append(activations, batch_activations.detach().cpu().numpy(), axis=0)
                
                #loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
            
                
#        with torch.no_grad():
#            _, activations, _ = model.forward(train_input_ids, train_input_mask, train_segment_ids)
        
        del train_features, train_input_ids, train_input_mask, train_segment_ids
        
        activations_=[]
        for t in activations:
            temp =[]
            for t1 in t:
                temp.append(t1)#t1.numpy()
            activations_.append(temp)
        
        activations = np.array(activations_)
        del activations_
        
        #print("ACTIVATION SHAPE:", activations.shape)#[10, 768]
        
        if isinstance(activations, np.ndarray):
            nodes_last_layer = np.shape(activations)[1]
        else:
            raise ValueError("activations is None or tensor.")

        if nodes_last_layer <= self.TOO_SMALL_ACTIVATIONS:
            logger.warning(
                "Number of activations in last hidden layer is too small. Method may not work properly. " "Size: %s",
                str(nodes_last_layer),
            )
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        return activations

    def _segment_by_class(self, data: np.ndarray, features: np.ndarray) -> List[np.ndarray]:
        """
        Returns segmented data according to specified features.

        :param data: Data to be segmented.
        :param features: Features used to segment data, e.g., segment according to predicted label or to `y_train`.
        :return: Segmented data according to specified features.
        """
        n_classes = self.classifier.num_labels
        return segment_by_class(data, features, n_classes)
    
    def cluster_activations(self, x_train, y_train, **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Clusters activations and returns cluster_by_class and red_activations_by_class, where cluster_by_class[i][j] is
        the cluster to which the j-th data point in the ith class belongs and the correspondent activations reduced by
        class red_activations_by_class[i][j].

        :param kwargs: A dictionary of cluster-specific parameters.
        :return: Clusters per class and activations by class.
        """
        self.set_params(**kwargs)
        #print("In cluster activation")
        
        if self.generator is not None:
            batch_size = self.generator.batch_size
            num_samples = self.generator.size
            num_classes = self.classifier.num_labels
            for batch_idx in range(num_samples // batch_size):  # type: ignore
                x_batch, y_batch = self.generator.get_batch()
                print("generating activation cluster_analysis batch generator")
                batch_activations = self._get_activations(self.classifier, x_batch, y_batch, max_seq_length = self.max_seq_length, tokenizer =self.tokenizer)
                activation_dim = batch_activations.shape[-1]

                # initialize values list of lists on first run
                if batch_idx == 0:
                    self.activations_by_class = [np.empty((0, activation_dim)) for _ in range(num_classes)]
                    clusters_by_class = [np.empty(0, dtype=int) for _ in range(num_classes)]
                    red_activations_by_class = [np.empty((0, self.nb_dims)) for _ in range(num_classes)]

                _activations_by_class = self._segment_by_class(batch_activations, y_batch)
                _clusters_by_class, _red_activations_by_class = cluster_activations(
                    activations_by_class,
                    nb_clusters=self.nb_clusters,
                    nb_dims=self.nb_dims,
                    reduce=self.reduce,
                    clustering_method=self.clustering_method,
                    generator=self.generator,
                    clusterer_new=self.clusterer,
                )

                for class_idx in range(num_classes):
                    self.activations_by_class[class_idx] = np.vstack(
                        [self.activations_by_class[class_idx], _activations_by_class[class_idx]]
                    )
                    clusters_by_class[class_idx] = np.append(
                        clusters_by_class[class_idx], _clusters_by_class[class_idx]
                    )
                    red_activations_by_class[class_idx] = np.vstack(
                        [red_activations_by_class[class_idx], _red_activations_by_class[class_idx]]
                    )
                del _activations_by_class, _clusters_by_class, _red_activations_by_class
            return clusters_by_class, red_activations_by_class

        if self.activations_by_class == []:
            print("generating activation cluster_analysis")
            activations = self._get_activations(self.classifier, x_test, y_test, max_seq_length = self.max_seq_length, tokenizer =self.tokenizer)
            self.activations_by_class = self._segment_by_class(activations, y_train)
            del activations

        [clusters_by_class, red_activations_by_class] = cluster_activations(
            self.activations_by_class,
            nb_clusters=self.nb_clusters,
            nb_dims=self.nb_dims,
            reduce=self.reduce,
            clustering_method=self.clustering_method,
        )

        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        return clusters_by_class, red_activations_by_class


def cluster_activations(
    separated_activations: List[np.ndarray],
    nb_clusters: int = 2,
    nb_dims: int = 10,
    reduce: str = "FastICA",
    clustering_method: str = "KMeans",
    generator: Optional[DataGenerator] = None,
    clusterer_new: Optional[MiniBatchKMeans] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Clusters activations and returns two arrays.
    1) separated_clusters: where separated_clusters[i] is a 1D array indicating which cluster each data point
    in the class has been assigned.
    2) separated_reduced_activations: activations with dimensionality reduced using the specified reduce method.

    :param separated_activations: List where separated_activations[i] is a np matrix for the ith class where
           each row corresponds to activations for a given data point.
    :param nb_clusters: number of clusters (defaults to 2 for poison/clean).
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :param clustering_method: Clustering method to use, default is KMeans.
    :param generator: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations).
    :param clusterer_new: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations)
    """
    #print("In second cluster_activations")
    separated_clusters = []
    separated_reduced_activations = []

    if clustering_method == "KMeans":
        clusterer = KMeans(n_clusters=self.nb_clusters, init="k-means++", n_init=10, random_state=45, max_iter=3000, verbose=0)
    else:
        raise ValueError(f"{clustering_method} clustering method not supported.")

    for activation in separated_activations:
        print("Apply dimensionality reduction")#, type(activation))
        
        nb_activations = len(activation[0])#np.shape(activation)[1]
        if nb_activations > nb_dims:
            # TODO: address issue where if fewer samples than nb_dims this fails
            reduced_activations = reduce_dimensionality(activation, nb_dims=nb_dims, reduce=reduce)
        else:
            logger.info(
                "Dimensionality of activations = %i less than nb_dims = %i. Not applying dimensionality " "reduction.",
                nb_activations,
                nb_dims,
            )
            reduced_activations = activation
        separated_reduced_activations.append(reduced_activations)

        # Get cluster assignments
        if generator is not None and clusterer_new is not None:
            clusterer_new = clusterer_new.partial_fit(reduced_activations)
            # NOTE: this may cause earlier predictions to be less accurate
            clusters = clusterer_new.predict(reduced_activations)
        else:
            clusters = clusterer.fit_predict(reduced_activations)
        separated_clusters.append(clusters)
        
    return separated_clusters, separated_reduced_activations

def measure_misclassification(
    classifier: "CLASSIFIER_NEURALNETWORK_TYPE", x_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Computes 1-accuracy given x_test and y_test

    :param classifier: Classifier to be used for predictions.
    :param x_test: Test set.
    :param y_test: Labels for test set.
    :return: 1-accuracy.
    """
    predictions = np.argmax(classifier.predict(x_test), axis=1)
    return 1.0 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]


def train_remove_backdoor(
    classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    tolerable_backdoor: float,
    max_epochs: int,
    batch_epochs: int,
) -> tuple:
    """
    Trains the provider classifier until the tolerance or number of maximum epochs are reached.

    :param classifier: Classifier to be used for predictions.
    :param x_train: Training set.
    :param y_train: Labels used for training.
    :param x_test: Samples in test set.
    :param y_test: Labels in test set.
    :param tolerable_backdoor: Parameter that determines how many misclassifications are acceptable.
    :param max_epochs: maximum number of epochs to be run.
    :param batch_epochs: groups of epochs that will be run together before checking for termination.
    :return: (improve_factor, classifier).
    """
    # Measure poison success in current model:
    initial_missed = measure_misclassification(classifier, x_test, y_test)

    curr_epochs = 0
    curr_missed = 1.0
    while curr_epochs < max_epochs and curr_missed > tolerable_backdoor:
        classifier.fit(x_train, y_train, nb_epochs=batch_epochs)
        curr_epochs += batch_epochs
        curr_missed = measure_misclassification(classifier, x_test, y_test)
        logger.info("Current epoch: %s", curr_epochs)
        logger.info("Misclassifications: %s", curr_missed)

    improve_factor = initial_missed - curr_missed
    return improve_factor, classifier





def reduce_dimensionality(activations: np.ndarray, nb_dims: int = 10, reduce: str = "FastICA") -> np.ndarray:
    """
    Reduces dimensionality of the activations provided using the specified number of dimensions and reduction technique.

    :param activations: Activations to be reduced.
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :return: Array with the reduced activations.
    """
    # pylint: disable=E0001
    from sklearn.decomposition import FastICA, PCA

    if reduce == "FastICA":
        projector = FastICA(random_state = 45, n_components=nb_dims, max_iter=1000, tol=0.005)
    elif reduce == "PCA":
        projector = PCA(n_components=nb_dims,random_state=45 )
    else:
        raise ValueError(f"{reduce} dimensionality reduction method not supported.")
    
    #print("Applying ", projector)

    #activations = [t.numpy() for t in activations]
    reduced_activations = projector.fit_transform(activations)
    return reduced_activations
