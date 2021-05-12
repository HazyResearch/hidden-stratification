from collections import defaultdict
from copy import deepcopy
import json
import logging
import os

import numpy as np
import torch

from stratification.cluster.models.cluster import DummyClusterer
from stratification.cluster.utils import (
    get_cluster_composition,
    get_cluster_mean_loss,
    get_k_from_model,
)
from stratification.utils.logger import init_logger


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
                metric = get_cluster_composition(inputs['true_subclass'], assignments)
            else:
                raise KeyError(f'Unrecognized metric_type {metric_type}')
            metrics[metric_type] = metric
        return metrics

    def train(self, cluster_model, inputs):
        """Fits cluster models to the data of each superclass.

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

        Returns:
            group_to_models(List[Tuple[type(cluster_model), type(reduction_model)]]): the list
                of reduction and cluster models fit on each group, where the idx
                indicates the group.
        """
        orig_cluster_model = cluster_model
        extra_info = hasattr(cluster_model, 'requires_extra_info')

        inputs_tr = inputs['train']
        inputs_val = inputs['val']

        group_to_models = []
        for group, group_data in inputs_tr[0].items():
            if group in self.config['superclasses_to_ignore']:
                # Keep this superclass in a single "cluster"
                self.logger.basic_info(f'Not clustering superclass {group}...')
                group_to_models.append(DummyClusterer())
                continue

            cluster_model = deepcopy(orig_cluster_model)
            activations = group_data['activations']

            if extra_info:
                val_group_data = inputs_val[0][group]
                losses = group_data['losses']
                val_activations = val_group_data['activations']
                kwargs = {'val_activ': val_activations, 'losses': losses}
            else:
                kwargs = {}

            # cluster
            self.logger.basic_info(f'Clustering superclass {group}...')
            cluster_model = cluster_model.fit(activations, **kwargs)
            group_to_models.append(cluster_model)

        return group_to_models

    def evaluate(self, group_to_models, split_inputs):
        """Returns cluster assignments for each of the inputs.

        Args:
            group_to_models(List[reduction_model]):
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
        group_to_data, group_assignments = split_inputs
        group_to_metrics = {}
        group_to_outputs = {}
        cluster_floor = 0
        for group, group_data in group_to_data.items():
            self.logger.info(f'Evaluating group {group}...')

            group_outputs = group_data.copy()
            cluster_model = group_to_models[group]
            assignments = np.array(cluster_model.predict(group_data['activations']))
            group_outputs['assignments'] = cluster_floor + assignments

            group_to_outputs[group] = group_outputs
            group_to_metrics[group] = self.compute_metrics(group_data, assignments)

            # update cluster_floor to ensure disjoint assignments
            k = get_k_from_model(cluster_model)  # accounts for degenerate cases
            cluster_floor = cluster_floor + k

        outputs = self._ungroup(group_to_outputs, group_assignments)
        return group_to_metrics, outputs

    def _ungroup(self, group_to_data, group_assignments):
        """Ungroups data that is partitioned by group.

        Args:
            group_to_data(Dict[int, Dict[str, Sequence]]) a partitioned
                group of data, i.e. the object returned by GEORGEReduce._group
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
