from copy import deepcopy
import os
from collections import Counter, defaultdict
import json
import logging

import torch
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

import stratification.cluster.models.reduction as reduction_models
import stratification.cluster.models.cluster as cluster_models
from stratification.cluster.utils import get_cluster_mean_loss, get_cluster_composition, get_k_from_model

from stratification.utils.logger import init_logger
from stratification.utils.utils import NumpyEncoder


class GEORGECluster:
    """Executes the cluster stage of the GEORGE algorithm.
    
    Args:
        cluster_config(dict): Contains the parameters required to execute this step.
            See utils.schema for type information and examples.
        save_dir(str, optional): Directory at which to save logging information.
            If None, logging information is not saved. Default is None.
    """
    def __init__(self, cluster_config, save_dir=None, log_format='full'):
        self.config = cluster_config
        self.save_dir = save_dir
        if self.save_dir:
            self.logger = init_logger('harness.cluster', self.save_dir, log_format=log_format)
        else:
            self.logger = logging.getLogger()

    def preprocess_activations(self, activations):
        """Preprocesses the activations based on keys in the config.
        
        Args:
            activations(np.ndarray of shape (N, D)): D-dimensional vectors for N
                samples.
        
        Returns:
            activations(np.ndarray of shape (N, D)): transformed activations.
        """
        if len(activations.shape) > 2:
            activations = activations.reshape(activations.shape[0], -1)
        if self.config['normalize']:
            # divide activations by their norm, with a lower threshold for numerical stability
            act_norms = np.maximum(np.linalg.norm(activations, axis=-1, keepdims=True), 1e-6)
            activations = activations / act_norms
        return activations

    def compute_metrics(self, inputs, assignments):
        """Computes metrics using the sample data provided in inputs.

        Args:
            inputs(Dict[str, Sequence]) inputs of the same format as 
                those described in GEORGECluster.train
            assignments(Sequence) the cluster assignments for each input

        Returns:
            metrics(Dict[str, Union[Dict[str, Any], float]]): the metrics computed.
                Can be per-cluster metrics or aggregate metrics.
        """
        metrics = {}
        for metric_type in self.config['metric_types']:
            if metric_type == 'mean_loss':
                metric = get_cluster_mean_loss(inputs['losses'], assignments)
            elif metric_type == 'composition':
                metric = get_cluster_composition(inputs['superclass'], assignments)
            else:
                raise KeyError(f'Unrecognized metric_type {metric_type}')
            metrics[metric_type] = metric
        return metrics

    def train(self, cluster_model, inputs, reduction_model=None):
        """Fits reduction and cluster models to the data.
        
        Note:
            It is possible to create groups of inputs (see 
            GEORGECluster.get_groups_as_list). Doing so causes G reduction and cluster
            models to be instantiated, where G is the number of groups created.
            At time of implementation, all resulting cluster models will have the same
            hyperparameters (the same is true for the reduction model).
        
        Args:
            cluster_model(Any): The model used to produce cluster assignments. Must
                implement `fit` and `predict`. Further, the number of clusters the 
                cluster_model will attempt to fit must be accessible, through either 
                (1) `n_clusters` or (2) `n_components`. This is due to the
                limitations of the sklearn implementations of KMeans and GMMs.
            inputs(Dict[str, Sequence]): a dictionary object containing the model
                activations and various metadata. The complete schema is the following:
                {
                    'metrics': Dict[str, Any],
                    'activations': np.ndarray of shape (N, D),
                    'superclass': np.ndarray of shape (N, ),
                    'subclass': np.ndarray of shape (N, ),
                    'true_subclass': np.ndarray of shape (N, ),
                    'targets': np.ndarray of shape (N, ),
                    'probs': np.ndarray of shape (N, ),
                    'preds': np.ndarray of shape (N, ),
                    'losses': np.ndarray of shape (N, ), 
                }
                Future work is to further modularize the cluster code to mitigate
                dependencies on this object. For best results, train classifiers
                using GEORGEHarness.classify.
            reduction_model(Any, optional): The model used for dimensionality reduction
                of the activations. If None, defaults to a NoOpReductionModel, which 
                simply returns the activations.
            
        Returns:
            group_to_models(List[Tuple[type(cluster_model), type(reduction_model)]]): the list
                of reduction and cluster models fit on each group, where the idx
                indicates the group.
        """
        if reduction_model is None:
            self.logger.basic_info('Using raw features (no dimensionality reduction)')
            reduction_model = NoOpReductionModel()

        group_assignments = self.get_groups_as_list(inputs)
        group_to_data = self._group(inputs, group_assignments)
        groups = np.unique(group_assignments)

        group_to_models = []
        for group in groups:
            group_data = group_to_data[group]
            cluster_model, reduction_model = deepcopy(cluster_model), deepcopy(reduction_model)

            # reduce
            self.logger.basic_info(f'Reducing superclass {group}...')
            activations = group_data['activations']
            activations = self.preprocess_activations(activations)
            acts_dtype = activations.dtype
            reduction_model = reduction_model.fit(activations)
            activations = reduction_model.transform(activations)
            activations = activations.astype(acts_dtype)

            # cluster
            self.logger.basic_info(f'Clustering superclass {group}...')
            cluster_model = cluster_model.fit(activations)
            group_to_models.append((cluster_model, reduction_model))

        self._save(group_to_models, 'clusters.pt')
        return group_to_models

    def evaluate(self, group_to_models, inputs):
        """Returns cluster assignments for each of the inputs.
        
        Args:
            group_to_models(List[Tuple[type(cluster_model), type(reduction_model)]]):
                the models produced by GEORGECluster.train. There should be as many
                items in this list as groups in the inputs.
            inputs(Dict[str, Sequence]): inputs of the same format as those described in
                GEORGECluster.train
        
        Returns:
            group_to_metrics(Dict[str, Any]): metrics, partitioned by group.
            outputs(Dict[str, Any]): the outputs of the model. At time of writing, 
                the outputs consists of both the reduced activations and the cluster
                assignments (`activations` and `assignments` keys, respectively).
        """
        group_assignments = self.get_groups_as_list(inputs)
        group_to_data = self._group(inputs, group_assignments)
        groups = np.unique(group_assignments)
        assert len(group_to_models) <= len(groups), \
            'There must be a model for each group in the input data.'

        group_to_metrics = {}
        group_to_outputs = {}
        cluster_floor = 0
        for group in groups:
            self.logger.info(f'Evaluating group {group}...')
            group_outputs = {}
            group_data = group_to_data[group]
            cluster_model, reduction_model = group_to_models[group]

            # reduce
            activations = group_data['activations']
            activations = self.preprocess_activations(activations)
            acts_dtype = activations.dtype
            activations = reduction_model.transform(activations)
            activations = activations.astype(acts_dtype)
            group_outputs['activations'] = activations

            # cluster
            assignments = np.array(cluster_model.predict(activations))
            group_outputs['assignments'] = cluster_floor + assignments

            group_to_outputs[group] = group_outputs
            group_to_metrics[group] = self.compute_metrics(group_data, assignments)

            # update cluster_floor to ensure disjoint assignments
            k = get_k_from_model(cluster_model)  # accounts for degenerate cases
            cluster_floor = cluster_floor + k

        outputs = self._ungroup(group_to_outputs, group_assignments)
        return group_to_metrics, outputs

    def get_groups_as_list(self, data):
        """Returns the group assignments of data"""
        if self.config['cluster_by_superclass']:
            self.logger.info('Grouping samples by given superclass...')
            group_assignments = data['superclass']
        else:
            self.logger.info('Treating samples as one group...')
            group_assignments = np.zeros_like(data['superclass'])
        return group_assignments

    def _group(self, data, group_assignments):
        """Partitions the data by group.
        
        Note: 
            this function assumes that the data is a dictionary of sequences.
            By design, any key-value pair that doesn't describe a sequence is
            ignored in the final partition.
        
        Args:
            data(Dict[str, Sequence]): A dictionary of sequences with the same
                length of `group_assignments`.
            group_assignments(Sequence[int]): A list of assignments of data,
                where `group_assignments[idx]` is the group of data[idx].
            
        Returns:
            group_to_data(Dict[int, Dict[str, Sequence]]): the data, partitioned by group.
                Note that the grouped data is still in the same order as it
                was before partitioning.
        """
        groups = np.unique(group_assignments)
        group_to_data = defaultdict(dict)
        for group in groups:
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    assert len(group_assignments) == len(v), \
                        f'group_assignments and "{k}" must be the same length'
                    group_to_data[group][k] = v[group_assignments == group]
        return group_to_data

    def _ungroup(self, group_to_data, group_assignments):
        """Ungroups data that is partitioned by group.
        
        Args:
            group_to_data(Dict[int, Dict[str, Sequence]]) a partitioned
                group of data, likely the object returned by GEORGECluster._group
            group_assignments(Sequence[int]): A list of assignments of data,
                where `group_assignments[idx]` is the group of data[idx].
        
        Returns:
            data(Dict[str, Sequence]): A dictionary of sequences with the same
                length of `group_assignments`.
        """
        # keep track of where we are in each group of group_to_data
        group_to_ptr = {group: 0 for group in group_to_data.keys()}
        data = defaultdict(list)
        for group in group_assignments:
            group_data = group_to_data[group]
            for k, v in group_data.items():
                data[k].append(v[group_to_ptr[group]])
            group_to_ptr[group] += 1

        # format
        for k, v in data.items():
            data[k] = np.array(v)
        return data

    def _save(self, data, filename):
        """If self.save_dir is not None, saves `data`."""
        if self.save_dir is not None:
            filepath = os.path.join(self.save_dir, filename)
            self.logger.basic_info(f'Saving checkpoint to {filepath}...')
            torch.save(data, filepath)
        else:
            self.logger.basic_info('save_dir not initialized. Skipping save step.')
