schema = {
    'type': 'object',
    'required': [
        'exp_dir',
        'mode',
        'dataset',
        'classification_config',
        'reduction_config',
        'cluster_config',
    ],
    'properties': {
        'seed': {'type': 'number', 'default': -1},
        'deterministic': {'type': 'boolean', 'default': True},
        'use_cuda': {'type': 'boolean', 'default': True},
        'allow_multigpu': {'type': 'boolean', 'default': False},
        'exp_dir': {'type': 'string', 'default': 'checkpoints/_debug'},
        'mode': {
            'type': 'string',
            'default': 'erm',  # choices: erm, superclass_gdro, true_subclass_gdro, random_gdro, george
        },
        'dataset': {
            'type': 'string',
            'default': 'mnist',  # choices: celeba, isic, mnist, waterbirds
        },
        'activations_dir': {'type': 'string', 'default': 'NONE'},
        'representation_dir': {'type': 'string', 'default': 'NONE'},
        'cluster_dir': {'type': 'string', 'default': 'NONE'},
        'classification_config': {
            'type': 'object',
            'required': [
                'model',
                'metric_types',
                'checkpoint_metric',
                'eval_only',
                'num_epochs',
                'batch_size',
                'criterion_config',
                'optimizer_config',
                'scheduler_config',
                'dataset_config',
            ],
            'properties': {
                'model': {'type': 'string', 'default': 'lenet4'},
                'erm_config': {'type': 'object', 'default': {}},
                'gdro_config': {'type': 'object', 'default': {}},
                'metric_types': {'type': 'array', 'examples': [['acc', 'loss']]},
                'checkpoint_metric': {
                    'type': 'string',
                    'examples': ['train_acc', 'train_loss', 'val_acc', 'val_loss'],
                },
                'eval_only': {'type': 'boolean', 'default': False},
                'eval_mode': {
                    'type': 'string',
                    'default': 'best',
                    'examples': [
                        'best',
                        'best_val_acc',
                        'best_val_subclass_rob_acc',
                        'best_val_acc_rw',
                        'best_val_subclass_rob_acc_rw',
                        'best_val_true_subclass_rob_acc',
                        'best_val_auroc',
                        'best_val_subclass_rob_auroc',
                        'best_val_true_subclass_rob_auroc',
                        'best_val_alt_subclass_rob_auroc',
                    ],
                },
                'save_act_only': {'type': 'boolean', 'default': False},
                'ban_reweight': {'type': 'boolean', 'default': False},
                'bit_pretrained': {'type': 'boolean', 'default': False},
                'num_epochs': {'type': 'number', 'default': 20},
                'workers': {'type': 'number', 'default': 8},
                'dataset_config': {'type': 'object', 'properties': {}},
                'criterion_config': {
                    'type': 'object',
                    'properties': {
                        'robust_lr': {'type': 'number', 'default': 0.01},
                        'stable_dro': {'type': 'boolean', 'default': True},
                        'size_adjustment': {'type': 'number', 'default': 0},
                        'auroc_gdro': {'type': 'boolean', 'default': False},
                    },
                },
                'optimizer_config': {
                    'type': 'object',
                    'required': ['class_args', 'class_name'],
                    'properties': {
                        'class_args': {
                            'type': 'object',
                            'examples': [{'lr': 2e-3, 'weight_decay': 1e-5}],
                        },
                        'class_name': {'type': 'string', 'examples': ['Adam']},
                    },
                },
                'scheduler_config': {
                    'type': 'object',
                    'required': ['class_args', 'class_name'],
                    'properties': {
                        'class_args': {'type': 'object', 'examples': [{'milestones': [50, 75]}]},
                        'class_name': {'type': 'string', 'examples': ['MultiStepLR']},
                    },
                },
                'show_progress': {'type': 'boolean', 'default': True},
                'reset_model_state': {'type': 'boolean', 'default': True},
                'save_every': {'type': 'number', 'default': -1},
            },
        },
        'reduction_config': {
            'type': 'object',
            'required': ['model'],
            'properties': {
                'model': {
                    'type': 'string',
                    'default': 'umap',  # choices: "none", "pca", "umap", "hardness"
                },
                'components': {'type': 'number', 'default': 2},
                'normalize': {'type': 'boolean', 'default': True},
                'mean_reduce': {'type': 'boolean', 'default': False},
            },
        },
        'cluster_config': {
            'type': 'object',
            'required': ['model', 'metric_types'],
            'properties': {
                'model': {'type': 'string', 'default': 'gmm'},  # choices: "gmm", "kmeans"
                'metric_types': {'type': 'array', 'examples': [['mean_loss', 'composition']]},
                'search_k': {'type': 'boolean', 'default': False},
                'k': {'type': 'number', 'default': 10},
                'sil_cuda': {'type': 'boolean', 'default': False},
                'overcluster': {'type': 'boolean', 'default': False},
                'overcluster_factor': {'type': 'number', 'default': 5},
                'superclasses_to_ignore': {
                    'type': 'array',
                    'items': {'type': 'number'},
                    'default': [],
                },
            },
        },
    },
}
