import os
import json
import time
from copy import deepcopy

import torch
import numpy as np

from stratification.utils.utils import NumpyEncoder, get_unique_str, keys_to_strings, merge_dicts
from stratification.utils.logger import init_logger
from stratification.utils.visualization import visualize_clusters_by_group

from stratification.classification.datasets import DATA_SPLITS
from stratification.classification.george_classification import GEORGEClassification
from stratification.cluster.george_cluster import GEORGECluster
from stratification.cluster.utils import get_k_from_model


class GEORGEHarness:
    """Training harness for the GEORGE algorithm. 
    
    Note:
        Users can execute all (using `run`) or some (using `classify` and `cluster`) 
        parts of the algorithm. The harness is meant to facilitate interactions with
        GEORGEClassification and GEORGECluster– formatting outputs and saving artifacts.
        
    Args:
        exp_dir(str): the directory in which to store all experiment artifacts,
            including the metrics, cluster assignments, and visualizations.
        use_cuda(bool, optional): a flag describing whether or not to train on
            the GPU.
    """
    def __init__(self, exp_dir, use_cuda=False, log_format='full'):
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        self.log_format = log_format
        self.logger = init_logger('harness', self.exp_dir, log_format=log_format)
        self.use_cuda = use_cuda

    def run(self, config, dataloaders, model, cluster_model, activation_layer=None,
            reduction_model=None):
        """Runs all three stages of the GEORGE pipeline: ERM, cluster, DRO.
        
        Args:
            config(dict): contains nested classification_config and 
                cluster_config dictionaries.
            dataloaders(Dict[str, DataLoader]): a dictionary mapping a data split
                to its given DataLoader (note that all data splits in DATA_SPLITS 
                must be specified). More information can be found in
                classification.datasets.
            model(nn.Module): a PyTorch model.
            cluster_model(Any): a clustering model. Must implement `fit` and `predict`
                methods. For more details, see cluster.george_cluster.
            reduction_model(Any): a dimensionality reduction model. Must implement
                `fit` and `transform`. For more details, see cluster.george_cluster.
            
        Returns:
            outputs(List[str]): 
                contains the paths to the artifacts for each phase of the algorithm.
        """
        self._save_config(self.exp_dir, config, msg='Saving full config')
        outputs = []

        # (1) ERM
        self.logger.basic_info('Running ERM step (1/3)...')
        state_dict = deepcopy(model.state_dict())
        erm_dir = self.classify(config['classification_config'], model, dataloaders, 'erm')
        outputs.append(erm_dir)

        # (2) cluster
        self.logger.basic_info('Running cluster step (2/3)...')
        cluster_dir = self.cluster(config['cluster_config'], cluster_model,
                                   inputs_path=os.path.join(erm_dir, 'outputs.pt'),
                                   reduction_model=reduction_model)
        outputs.append(cluster_dir)

        # (3) DRO
        self.logger.basic_info(f'Running DRO step (3/3)...')
        if config['classification_config']['reset_model_state']:
            model.load_state_dict(state_dict)
            self.logger.basic_info('Model state reset')
        dro_dir = self.classify(config['classification_config'], model, dataloaders, 'george',
                                clusters_path=os.path.join(cluster_dir, 'clusters.pt'))
        outputs.append(dro_dir)
        return outputs

    def classify(self, classification_config, model, dataloaders, mode, clusters_path=None):
        """Runs the classification stage of the GEORGE pipeline.

        Note:
            This function handles much of the pre- and post-processing needed to transition
            from stage to stage (i.e. modifying datasets with subclass labels and formatting
            the outputs in a manner that is compatible with GEORGEHarness.cluster).
            For more direct interaction with the classification procedure, see the
            GEORGEClassification class in classification.george_classification.
        
        Args:
            classification_config(dict): Contains args for the criterion, optimizer, 
                scheduler, metrics. Optional nested `{mode}_config` dictionaries can
                add and/or replace arguments in classification config.
            model(nn.Module): A PyTorch model.
            dataloaders(Dict[str, DataLoader]): a dictionary mapping a data split
                to its given DataLoader (note that all data splits in DATA_SPLITS 
                must be specified). More information can be found in
                classification.datasets.
            mode(str): The type of optimization to run. `erm` trains with vanilla 
                Cross Entropy Loss. 'george' trains DRO using given cluster labels.
                `random_dro` trains DRO with random cluster labels. `superclass_dro`
                trains DRO using the given superclass labels. Implementation of DRO
                from Sagawa et al. (2020).
            clusters_path(str, optional): The path leading to clusters.pt file 
                produced by GEORGEHarness.cluster. Only needed if mode == 'george'.
                
        Returns:
            save_dir(str): The subdirectory within `exp_dir` that contains model 
                checkpoints, best model outputs, and best model metrics.
        """
        # overwrite args in classification_config with mode specific params
        mode_config = classification_config[f'{mode}_config']
        classification_config = merge_dicts(classification_config, mode_config)

        save_dir = os.path.join(self.exp_dir, f'{mode}_{get_unique_str()}')
        self._save_config(save_dir, classification_config)

        # extract subclass labels
        subclass_labels = self._get_subclass_labels(mode, clusters_path)
        for split, dataloader in dataloaders.items():
            key = 'train' if split == 'train_clean' else split
            if subclass_labels is not None:
                split_subclass_labels = subclass_labels[key]
            else:
                split_subclass_labels = None
            self._add_subclass_labels(dataloader, split_subclass_labels, mode)
        robust = self._get_robust_status(mode)

        # (1) train
        trainer = GEORGEClassification(
            classification_config, save_dir=save_dir, use_cuda=self.use_cuda,
            log_format=self.log_format, has_estimated_subclasses=mode
            not in ['erm', 'true_subclass_dro'])
        if not classification_config['eval_only']:
            trainer.train(model, dataloaders['train'], dataloaders['val'], robust=robust)

        # (2) evaluate
        split_to_outputs = {}
        split_to_metrics = {}
        for split, dataloader in dataloaders.items():
            if split == 'train':
                continue
            key = 'train' if split == 'train_clean' else split
            self.logger.basic_info(f'Evaluating on {key} split...')
            metrics, outputs = trainer.evaluate(model, dataloader, robust=robust,
                                                save_activations=True)
            split_to_metrics[key] = metrics
            split_to_outputs[key] = outputs

        # (3) save everything
        self._save_json(os.path.join(save_dir, 'metrics.json'), split_to_metrics)
        self._save_torch(os.path.join(save_dir, 'outputs.pt'), split_to_outputs)
        return save_dir

    def cluster(self, cluster_config, cluster_model, inputs_path, reduction_model=None):
        """
        Runs clustering stage of the GEORGE pipeline.
        
        Note:
            The `inputs_path` parameter must describe a pickle-serialized dictionary
            that has the following schema:
            {
                'train': {
                    'metrics': Dict[str, Any],
                    'activations': np.ndarray of shape (N, D),
                    'superclass': np.ndarray of shape (N, ),
                    'subclass': np.ndarray of shape (N, ),
                    'true_subclass': np.ndarray of shape (N, ),
                    'targets': np.ndarray of shape (N, ),
                    'probs': np.ndarray of shape (N, ),
                    'preds': np.ndarray of shape (N, ),
                    'losses': np.ndarray of shape (N, ), 
                },
                'val': {...},
                'test': {...}
            }
            Future work is to further modularize the cluster code to mitigate
            dependencies on this object. For best results, train classifiers
            using GEORGEHarness.classify.
            
        Args:
            cluster_config(dict): contains args for the clustering step.
            cluster_model(Any): a clustering model. Must implement `fit` and `predict`
                methods. For more details, see cluster.george_cluster.
            inputs_path (str) path leading to outputs.pt file produced by 
                GEORGEHarness.classify.
            reduction_model(Any): a dimensionality reduction model. Must implement
                `fit` and `transform`. For more details, see cluster.george_cluster.
                
        Returns:
            save_dir(str). subdirectory within `exp_dir` that contains the cluster
                assignments, other cluster output, and cluster metrics.
        """
        save_dir = os.path.join(os.path.dirname(inputs_path), f'cluster_{get_unique_str()}')
        self._save_config(save_dir, cluster_config, msg='Saving cluster step config')
        inputs = torch.load(inputs_path)
        assert len(set(inputs.keys()) & {'train', 'val', 'test'}) == 3, \
            'Must have ["train", "val", "test"] splits.'
        for split, split_inputs in inputs.items():
            assert len(set(split_inputs.keys()) & {'superclass', 'activations'}) == 2, \
                f'{split} split of loaded inputs must have ["superclass", "activations"] keys'

        # (1) train
        trainer = GEORGECluster(cluster_config, save_dir=save_dir, log_format=self.log_format)
        group_to_models = trainer.train(cluster_model, inputs['train'],
                                        reduction_model=reduction_model)

        # (2) evaluate
        split_to_metrics = {}
        split_to_outputs = {}
        for split, split_inputs in inputs.items():
            metrics, outputs = trainer.evaluate(group_to_models, inputs[split])
            split_to_metrics[split] = metrics
            split_to_outputs[split] = outputs

        # (3) save everything
        self._save_cluster_visualizations(save_dir, inputs, group_to_models, split_to_outputs,
                                          trainer)
        self._save_json(os.path.join(save_dir, 'metrics.json'), split_to_metrics)
        self._save_torch(os.path.join(save_dir, 'outputs.pt'), split_to_outputs)
        # save assignments only
        split_to_assignments = {k: v['assignments'] for k, v in split_to_outputs.items()}
        self._save_torch(os.path.join(save_dir, 'clusters.pt'), split_to_assignments)
        return save_dir

    def _get_robust_status(self, mode):
        """Identifies if the given `mode` calls for DRO"""
        if mode in {'george', 'random_dro', 'superclass_dro', 'true_subclass_dro'}:
            return True
        elif mode == 'erm':
            return False
        raise ValueError('mode {mode} not valid. Use one of the following:\n' +
                         '["george", "random_dro", "superclass_dro", "true_subclass_dro", ' +
                         '"erm"]')

    def _get_subclass_labels(self, mode, clusters_path):
        """Gets subclass labels if `mode` is 'george'."""
        if mode == 'george':
            if clusters_path is None:
                raise ValueError('to run george, clusters_path must be defined.')
            return torch.load(clusters_path)  # must be {split: labels} dict
        return None

    def _add_subclass_labels(self, dataloader, subclass_labels, mode):
        """Adds subclass labels to the dataset of a given dataloader
        
        Args:
            dataloader(DataLoader): a PyTorch DataLoader containing a GEORGEDataset.
            subclass_labels(Union[str, Sequence]): a description or an explicit
                sequence of subclass labels.
            mode(str): The type of optimization to run. Determines the type of subclass
                labels to pass into dataloader.dataset.
        """
        if mode == 'george':
            if subclass_labels is None:
                raise ValueError('subclass_labels must not be None if running `george`')
        elif mode == 'random_dro':
            subclass_labels = 'random'
        elif mode == 'superclass_dro':
            subclass_labels = 'superclass'
        elif mode == 'true_subclass_dro':
            subclass_labels = 'true_subclass'
        elif mode == 'erm':
            # subclass labels default to superclass labels if none are given
            subclass_labels = 'superclass'

        dataset = dataloader.dataset
        split = dataset.split
        self.logger.info(f'{split} split:')

        dataset.add_subclass_labels(subclass_labels)

        # log class counts for each label type
        for label_type, labels in dataset.Y_dict.items():
            self.logger.info(f'{label_type.capitalize()} counts: {np.bincount(labels)}')

    def _save_cluster_visualizations(self, save_dir, inputs, group_to_models, split_to_outputs,
                                     trainer):
        """Generates and saves cluster visualizations."""
        group_to_k = {
            group: get_k_from_model(cluster_model)
            for group, (cluster_model, _) in enumerate(group_to_models)
        }
        for split, outputs in split_to_outputs.items():
            visualization_dir = os.path.join(save_dir, 'visualizations', split)
            os.makedirs(visualization_dir, exist_ok=True)
            visualize_clusters_by_group(outputs['activations'], outputs['assignments'],
                                        trainer.get_groups_as_list(inputs[split]),
                                        true_subclass_labels=inputs[split]['true_subclass'],
                                        group_to_k=group_to_k, save_dir=visualization_dir)

    def _save_config(self, save_dir, config, msg=None):
        """Helper function to save `config` in `save_dir`."""
        os.makedirs(save_dir, exist_ok=True)
        self._save_json(os.path.join(save_dir, 'config.json'), config)
        if msg is not None:
            self.logger.info(msg)
        self.logger.basic_info(f'Config saved in: {save_dir}')
        self.logger.info(f'Config:\n{json.dumps(config, indent=4)}')

    def _save_json(self, save_path, data):
        """Saves JSON type objects."""
        with open(save_path, 'w') as f:
            json.dump(keys_to_strings(data), f, indent=4, cls=NumpyEncoder)

    def _save_torch(self, save_path, data):
        """Saves arbitrary data with the torch serializer."""
        torch.save(data, save_path)
