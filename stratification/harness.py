import logging
import os
import json
import time
from copy import deepcopy
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from stratification.utils.utils import NumpyEncoder, get_unique_str, keys_to_strings, merge_dicts
from stratification.utils.logger import init_logger
from stratification.utils.visualization import visualize_clusters_by_group

from stratification.classification.datasets import *
from stratification.classification.models import *
from stratification.cluster.models.cluster import *
from stratification.cluster.models.reduction import *

from stratification.classification.george_classification import GEORGEClassification
from stratification.cluster.george_reduce import GEORGEReducer
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
    def __init__(self, config, use_cuda=False, log_format='full'):
        if config['classification_config']['eval_only'] or config['classification_config'][
                'save_act_only']:
            self.exp_dir = config['activations_dir']
        else:
            self.exp_dir = os.path.join(config['exp_dir'], 'run_' + get_unique_str())
            os.makedirs(self.exp_dir, exist_ok=True)
        self.log_format = log_format
        self.logger = init_logger('harness', self.exp_dir, log_format=log_format)
        self.use_cuda = use_cuda

    def save_full_config(self, config):
        self._save_config(self.exp_dir, config, msg='Saving full config')

    def run_george(self, config, dataloaders, model, cluster_model, reduction_model,
                   activation_layer=None):
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
        self.save_full_config(config)
        outputs = []

        # (1) Train initial representation
        self.logger.basic_info('Training initial representation step (1/3)...')
        state_dict = deepcopy(model.state_dict())
        erm_dir = self.classify(config['classification_config'], model, dataloaders, 'erm')
        outputs.append(erm_dir)

        self.logger.basic_info('Running reduction step (2/3)...')
        reduction_dir = self.reduce(config, inputs_path=os.path.join(erm_dir, 'outputs.pt'))
        outputs.append(reduction_dir)

        # (2) cluster
        self.logger.basic_info('Running cluster step (2/3)...')
        cluster_dir = self.cluster(config, inputs_path=os.path.join(reduction_dir, 'outputs.pt'))
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

    def classify(self, classification_config, model, dataloaders, mode):
        """Runs the initial representation learning stage of the GEORGE pipeline.

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
                `random_gdro` trains DRO with random cluster labels. `superclass_gdro`
                trains DRO using the given superclass labels. Implementation of DRO
                from Sagawa et al. (2020).
            clusters_path(str, optional): The path leading to clusters.pt file 
                produced by GEORGEHarness.cluster. Only needed if mode == 'george'.
                
        Returns:
            save_dir(str): The subdirectory within `exp_dir` that contains model 
                checkpoints, best model outputs, and best model metrics.
        """
        # overwrite args in classification_config with mode specific params
        if mode == 'erm':
            mode_config = classification_config[f'erm_config']
        else:
            mode_config = classification_config[f'gdro_config']
        classification_config = merge_dicts(classification_config, mode_config)

        if classification_config['eval_only'] or classification_config['save_act_only']:
            save_dir = self.exp_dir
        else:
            save_dir = os.path.join(self.exp_dir, f'{mode}_{get_unique_str()}')
            self._save_config(save_dir, classification_config)
        robust = self._get_robust_status(mode)

        # (1) train
        trainer = GEORGEClassification(
            classification_config, save_dir=save_dir, use_cuda=self.use_cuda,
            log_format=self.log_format,
            has_estimated_subclasses=mode not in ['erm', 'true_subclass_gdro'])
        if not (classification_config['eval_only'] or classification_config['save_act_only']
                or classification_config['bit_pretrained']):
            trainer.train(model, dataloaders['train'], dataloaders['val'], robust=robust)

        # (2) evaluate
        split_to_outputs = {}
        split_to_metrics = {}
        for split, dataloader in dataloaders.items():
            if split == 'train':
                continue
            key = 'train' if split == 'train_clean' else split
            if classification_config['eval_only'] and key != 'test':
                continue
            self.logger.basic_info(f'Evaluating on {key} split...')
            metrics, outputs = trainer.evaluate(
                model, dataloaders, split, robust=robust, save_activations=True,
                bit_pretrained=classification_config['bit_pretrained'],
                adv_metrics=classification_config['eval_only'],
                ban_reweight=classification_config['ban_reweight'])
            split_to_metrics[key] = metrics
            split_to_outputs[key] = outputs

        # (3) save everything
        if not classification_config['eval_only']:
            self._save_json(os.path.join(save_dir, 'metrics.json'), split_to_metrics)
            self._save_torch(os.path.join(save_dir, 'outputs.pt'), split_to_outputs)
        return save_dir

    def reduce(self, reduction_config, reduction_model, inputs_path):
        save_dir = os.path.join(os.path.dirname(inputs_path), f'reduce_{get_unique_str()}')
        self._save_config(save_dir, reduction_config, msg='Saving reduction step config')
        inputs = torch.load(inputs_path)
        assert len(set(inputs.keys()) & {'train', 'val', 'test'}) == 3, \
            'Must have ["train", "val", "test"] splits.'
        for split, split_inputs in inputs.items():
            assert len(set(split_inputs.keys()) & {'superclass', 'activations'}) == 2, \
                f'{split} split of loaded inputs must have ["superclass", "activations"] keys'

        # apply dimensionality reduction (if specified) to the data
        reducer = GEORGEReducer(reduction_config, save_dir=save_dir, log_format=self.log_format)
        group_to_models, train_means = reducer.train(reduction_model, inputs)

        split_to_outputs = {}
        for split, split_inputs in inputs.items():
            outputs = reducer.evaluate(group_to_models, inputs[split], train_means)
            split_to_outputs[split] = (outputs, inputs[split]['superclass'])

        # save reduced data
        self._save_torch(os.path.join(save_dir, 'outputs.pt'), split_to_outputs)
        return save_dir

    def cluster(self, cluster_config, cluster_model, inputs_path):
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
            for group, group_data in split_inputs[0].items():
                assert len(set(group_data.keys()) & {'activations', 'losses'}) == 2, \
                    f'{split} split of loaded inputs must have ["activations", "losses"] keys' \
                     ' for each superclass'

        # (1) train
        c_trainer = GEORGECluster(cluster_config, save_dir=save_dir, log_format=self.log_format)
        group_to_models = c_trainer.train(cluster_model, inputs)

        # (2) evaluate
        split_to_metrics = {}
        split_to_outputs = {}
        for split, split_inputs in inputs.items():
            metrics, outputs = c_trainer.evaluate(group_to_models, inputs[split])
            split_to_metrics[split] = metrics
            split_to_outputs[split] = outputs

        # (3) save everything
        self._save_json(os.path.join(save_dir, 'metrics.json'), split_to_metrics)
        self._save_torch(os.path.join(save_dir, 'outputs.pt'), split_to_outputs)
        # save assignments only
        split_to_assignments = {k: v['assignments'] for k, v in split_to_outputs.items()}
        self._save_torch(os.path.join(save_dir, 'clusters.pt'), split_to_assignments)
        group_to_k = {
            group: get_k_from_model(cluster_model)
            for group, cluster_model in enumerate(group_to_models)
        }
        self._save_cluster_visualizations(save_dir, inputs, group_to_k, split_to_outputs, c_trainer)
        return save_dir

    def _get_robust_status(self, mode):
        """Identifies if the given `mode` calls for DRO"""
        if mode in {'george', 'random_gdro', 'superclass_gdro', 'true_subclass_gdro'}:
            return True
        elif mode == 'erm':
            return False
        raise ValueError('mode {mode} not valid. Use one of the following:\n' +
                         '["george", "random_gdro", "superclass_gdro", "true_subclass_gdro", ' +
                         '"erm"]')

    def get_dataloaders(self, config, mode='erm', transforms=None, subclass_labels=None):
        dataset_name = config['dataset']
        seed = config['seed']
        config = config['classification_config']
        if mode == 'george':
            assert ('.pt' in subclass_labels)  # file path to subclass labels specified
        elif mode != 'erm':
            assert (subclass_labels is None)
            subclass_labels = mode.rstrip('_gdro')

        if subclass_labels is None:
            # subclass labels default to superclass labels if none are given
            subclass_labels = 'superclass'

        if '.pt' in subclass_labels:  # file path specified
            subclass_labels = torch.load(subclass_labels)
        else:  # keyword specified
            kw = subclass_labels
            subclass_labels = defaultdict(lambda: kw)

        if mode == 'erm':
            mode_config = config[f'erm_config']
        else:
            mode_config = config[f'gdro_config']
        config = merge_dicts(config, mode_config)

        dataset_name = dataset_name.lower()
        d = {
            'celeba': CelebADataset,
            'isic': ISICDataset,
            'mnist': MNISTDataset,
            'waterbirds': WaterbirdsDataset
        }
        dataset_class = d[dataset_name]
        batch_size = config['batch_size']

        dataloaders = {}
        for split in DATA_SPLITS:
            key = 'train' if 'train' in split else split
            split_subclass_labels = subclass_labels[key]
            shared_dl_args = {'batch_size': batch_size, 'num_workers': config['workers']}
            if split == 'train':
                dataset = dataset_class(root='./data', split=split, download=True, augment=True,
                                        **config['dataset_config'])
                dataset.add_subclass_labels(split_subclass_labels, seed=seed)
                if config.get('uniform_group_sampling', False):
                    sampler, group_weights = self._get_uniform_group_sampler(dataset)
                    self.logger.info(
                        f'Resampling training data with subclass weights:\n{group_weights}')
                    dataloaders[split] = DataLoader(dataset, **shared_dl_args, shuffle=False,
                                                    sampler=sampler)
                else:
                    dataloaders[split] = DataLoader(dataset, **shared_dl_args, shuffle=True)
            else:
                # Evaluation dataloaders (including for the training set) are "clean" - no data augmentation or shuffling
                dataset = dataset_class(root='./data', split=key, **config['dataset_config'])
                dataset.add_subclass_labels(split_subclass_labels, seed=seed)
                dataloaders[split] = DataLoader(dataset, **shared_dl_args, shuffle=False)

            self.logger.info(f'{split} split:')
            # log class counts for each label type
            for label_type, labels in dataset.Y_dict.items():
                self.logger.info(f'{label_type.capitalize()} counts: {np.bincount(labels)}')

        return dataloaders

    def _get_uniform_group_sampler(self, dataset):
        group_counts, group_labels = dataset.get_class_counts('subclass'), dataset.get_labels(
            'subclass')
        group_weights = np.array([len(dataset) / c if c != 0 else 0 for c in group_counts])
        group_weights /= np.sum(group_weights)
        weights = group_weights[np.array(group_labels)]
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
        return sampler, group_weights

    def get_nn_model(self, config, num_classes, mode='erm'):
        cl_config = config['classification_config']
        if mode == 'erm':
            mode_config = cl_config[f'erm_config']
        else:
            mode_config = cl_config[f'gdro_config']
        cl_config = merge_dicts(cl_config, mode_config)
        if cl_config['bit_pretrained']:
            model_cls = BiTResNet
        else:
            models = {'lenet4': LeNet4, 'resnet50': PyTorchResNet, 'shallow_cnn': ShallowCNN}
            try:
                model_cls = models[cl_config['model']]
            except KeyError:
                raise ValueError('Unsupported model architecture')
        model = model_cls(num_classes=num_classes)
        if self.use_cuda:
            model = torch.nn.DataParallel(model).cuda()
        self.logger.info('Model:')
        self.logger.info(str(model))
        return model

    def get_reduction_model(self, config, nn_model=None):
        red_config = config['reduction_config']
        models = {
            'none': NoOpReducer,
            'pca': PCAReducer,
            'umap': UMAPReducer,
            'hardness': HardnessAugmentedReducer
        }
        if red_config['model'] != 'hardness':
            reduction_cls = models[red_config['model']]
            reduction_model = reduction_cls(random_state=config['seed'],
                                            n_components=red_config['components'])
        else:
            assert (nn_model is not None)
            base_reduction_model = UMAPReducer(random_state=config['seed'],
                                               n_components=red_config['components'])
            reduction_model = HardnessAugmentedReducer(nn_model, base_reduction_model)
        return reduction_model

    def get_cluster_model(self, config):
        cluster_config = config['cluster_config']

        kwargs = {
            'cluster_method': cluster_config['model'],
            'max_k': cluster_config['k'],
            'seed': config['seed'],
            'sil_cuda': cluster_config['sil_cuda'],
            'search': cluster_config['search_k']
        }
        if cluster_config['overcluster']:
            cluster_model = OverclusterModel(**kwargs, oc_fac=cluster_config['overcluster_factor'])
        else:
            cluster_model = AutoKMixtureModel(**kwargs)
        return cluster_model

    def _save_cluster_visualizations(self, save_dir, inputs, group_to_k, split_to_outputs, trainer):
        """Generates and saves cluster visualizations."""
        for split, outputs in split_to_outputs.items():
            visualization_dir = os.path.join(save_dir, 'visualizations', split)
            os.makedirs(visualization_dir, exist_ok=True)
            visualize_clusters_by_group(outputs['activations'],
                                        cluster_assignments=outputs['assignments'],
                                        group_assignments=inputs[split][1],
                                        true_subclass_labels=outputs['true_subclass'],
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
