from collections import defaultdict
from copy import deepcopy
import logging
import os

import numpy as np
import torch

import stratification.cluster.models.reduction as reduction_models
from stratification.utils.logger import init_logger


class GEORGEReducer:
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
            self.logger = init_logger('harness.reduction', self.save_dir, log_format=log_format)
        else:
            self.logger = logging.getLogger()

    def preprocess_activations(self, activations, means=None):
        """Preprocesses the activations based on keys in the config.

        Args:
            activations(np.ndarray of shape (N, D)): D-dimensional vectors for N
                samples.

        Returns:
            activations(np.ndarray of shape (N, D)): transformed activations.
        """
        if len(activations.shape) > 2:
            activations = activations.reshape(activations.shape[0], -1)
        if means is not None:
            activations = activations - means
        if self.config['normalize']:
            # divide activation vectors by their norm (plus epsilon, for numerical stability)
            act_norms = np.maximum(np.linalg.norm(activations, axis=-1, keepdims=True), 1e-6)
            activations = activations / act_norms
        return activations

    def train(self, reduction_model, inputs):
        """Fits reduction and cluster models to the data.
            'G' reduction and cluster models are instantiated, where G is the number
            of groups (i.e. superclasses).
            Currently, all resulting reduction and cluster models have the same
            hyperparameters for each superclass.

        Args:
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
            reduction_model(Any): The model used for dimensionality reduction
                of the activations.

        Returns:
            group_to_models(List[reduction_model]): the list of reduction models
                fit on each group, where the idx indicates the group.
        """
        orig_reduc_model = reduction_model

        inputs_tr = inputs['train']
        if 'losses' not in inputs_tr or len(inputs_tr['losses']) == 0:
            inputs_val = inputs['val']
            inputs_test = inputs['test']
            inputs_tr['losses'] = np.zeros(
                len(inputs_tr['activations']), dtype=inputs_tr['activations'].dtype
            )
            inputs_val['losses'] = np.zeros(
                len(inputs_val['activations']), dtype=inputs_val['activations'].dtype
            )
            inputs_test['losses'] = np.zeros(
                len(inputs_test['activations']), dtype=inputs_test['activations'].dtype
            )

        if self.config['mean_reduce']:
            train_means = (
                inputs_tr['activations']
                .reshape(inputs_tr['activations'].shape[0], -1)
                .mean(axis=0, keepdims=True)
            )
        else:
            train_means = None

        group_assignments = inputs_tr['superclass']
        group_to_data = self._group(inputs_tr, group_assignments)
        groups = np.unique(group_assignments)

        group_to_models = []
        for group in groups:
            group_data = group_to_data[group]
            reduction_model = deepcopy(orig_reduc_model)

            # reduce
            self.logger.basic_info(f'Fitting reduction model on superclass {group}...')
            activations = group_data['activations']
            if self.config['mean_reduce']:
                activations = self.preprocess_activations(activations, train_means)
            else:
                activations = self.preprocess_activations(activations)
            acts_dtype = activations.dtype
            reduction_model = reduction_model.fit(activations)
            activations = reduction_model.transform(activations)
            activations = activations.astype(acts_dtype)

            group_to_models.append(reduction_model)

        return group_to_models, train_means

    def evaluate(self, group_to_models, split_inputs, train_means=None):
        """Reduces each of the inputs.

        Args:
            group_to_models(List[reduction_model]):
                the models produced by GEORGEReduce.train. There should be as many
                items in this list as groups in the inputs.
            inputs(Dict[str, Sequence]): inputs of the same format as those described in
                GEORGEReduce.train

        Returns:
            group_to_metrics(Dict[str, Any]): metrics, partitioned by group.
            outputs(Dict[str, Any]): the outputs of the model. At time of writing,
                the outputs consists of both the reduced activations and the cluster
                assignments (`activations` and `assignments` keys, respectively).
        """
        if self.config['mean_reduce']:
            assert train_means is not None
        group_assignments = split_inputs['superclass']
        group_to_data = self._group(split_inputs, group_assignments)
        groups = np.unique(group_assignments)
        assert len(group_to_models) <= len(
            groups
        ), 'There must be a model for each group in the input data.'

        group_to_outputs = {}
        for group in groups:
            self.logger.info(f'Reducing group {group}...')
            group_data = group_to_data[group]
            group_outputs = group_data.copy()
            del group_outputs['superclass']  # unneeded, as all are superclass "group"
            reduction_model = group_to_models[group]

            # reduce
            activations = group_data['activations']
            if self.config['mean_reduce']:
                activations = self.preprocess_activations(activations, train_means)
            else:
                activations = self.preprocess_activations(activations)
            acts_dtype = activations.dtype
            activations = reduction_model.transform(activations)
            activations = activations.astype(acts_dtype)
            group_outputs['activations'] = activations

            group_to_outputs[group] = group_outputs

        return group_to_outputs

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
                    assert len(group_assignments) == len(
                        v
                    ), f'group_assignments and "{k}" must be the same length'
                    group_to_data[group][k] = v[group_assignments == group]
        return group_to_data
