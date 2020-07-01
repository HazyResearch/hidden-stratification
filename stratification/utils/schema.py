schema = {
    'type': 'object',
    'properties': {
        'seed': {
            'type': 'number',
            'default': 0
        },
        'exp_dir': {
            'type': 'string',
            'default': 'checkpoints/_debug'
        },
        'classification_config': {
            'type':
            'object',
            'required': [
                'metric_types',
                'checkpoint_metric',
                'eval_only',
                'num_epochs',
                'criterion_config',
                'optimizer_config',
                'scheduler_config',
            ],
            'properties': {
                'erm_config': {
                    'type': 'object',
                    'default': {}
                },
                'george_config': {
                    'type': 'object',
                    'default': {}
                },
                'random_dro_config': {
                    'type': 'object',
                    'default': {}
                },
                'superclass_dro_config': {
                    'type': 'object',
                    'default': {}
                },
                'true_subclass_dro_config': {
                    'type': 'object',
                    'default': {}
                },
                'metric_types': {
                    'type': 'array',
                    'examples': [['acc', 'loss']]
                },
                'checkpoint_metric': {
                    'type': 'string',
                    'examples': ['train_acc', 'train_loss', 'val_acc', 'val_loss']
                },
                'eval_only': {
                    'type': 'boolean',
                    'default': False
                },
                'num_epochs': {
                    'type': 'number',
                    'default': 20
                },
                'criterion_config': {
                    'type': 'object',
                    'properties': {
                        'robust_lr': {
                            'type': 'number',
                            'default': 0.01
                        },
                        'stable_dro': {
                            'type': 'boolean',
                            'default': True
                        },
                    }
                },
                'optimizer_config': {
                    'type': 'object',
                    'required': ['class_args', 'class_name'],
                    'properties': {
                        'class_args': {
                            'type': 'object',
                            'examples': [{
                                'lr': 2e-3,
                                'weight_decay': 1e-5
                            }]
                        },
                        'class_name': {
                            'type': 'string',
                            'examples': ['Adam']
                        }
                    }
                },
                'scheduler_config': {
                    'type': 'object',
                    'required': ['class_args', 'class_name'],
                    'properties': {
                        'class_args': {
                            'type': 'object',
                            'examples': [{
                                'milestones': [50, 75]
                            }]
                        },
                        'class_name': {
                            'type': 'string',
                            'examples': ['MultiStepLR']
                        }
                    }
                },
                'show_progress': {
                    'type': 'boolean',
                    'default': True
                },
                'reset_model_state': {
                    'type': 'boolean',
                    'default': True
                },
                'save_every': {
                    'type': 'number',
                    'default': -1
                }
            }
        },
        'cluster_config': {
            'type': 'object',
            'required': ['metric_types'],
            'properties': {
                'metric_types': {
                    'type': 'array',
                    'examples': [['mean_loss', 'composition']]
                },
                'cluster_by_superclass': {
                    'type': 'boolean',
                    'default': True
                },
                'normalize': {
                    'type': 'boolean',
                    'default': True
                }
            }
        }
    }
}
