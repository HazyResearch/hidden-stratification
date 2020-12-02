import os
import torch

from stratification.harness import GEORGEHarness
from stratification.utils.utils import set_seed, init_cuda
from stratification.utils.parse_args import get_config


def main():
    config = get_config()
    use_cuda = config['use_cuda'] and torch.cuda.is_available()
    set_seed(config['seed'], use_cuda)  # set seeds for reproducibility
    init_cuda(config['deterministic'], config['allow_multigpu'])

    torch.multiprocessing.set_sharing_strategy('file_system')
    harness = GEORGEHarness(config, use_cuda=use_cuda)
    harness.save_full_config(config)

    first_mode = 'erm' if (config['mode'] == 'george') else config['mode']
    dataloaders = harness.get_dataloaders(config, mode=first_mode)
    num_classes = dataloaders['train'].dataset.get_num_classes('superclass')
    model = harness.get_nn_model(config, num_classes=num_classes, mode=first_mode)

    activ_done = config['activations_dir'] != 'NONE'
    rep_done = config['representation_dir'] != 'NONE'
    cluster_done = config['cluster_dir'] != 'NONE'
    rep_done = rep_done or cluster_done  # if we already have clusters, don't need to do reduction step
    activ_done = activ_done or rep_done  # don't need to get activations if we already have reduced ones
    if config['classification_config']['eval_only']:
        assert (activ_done)
        if config['cluster_dir'] != 'NONE':
            dataloaders = harness.get_dataloaders(
                config, mode=first_mode,
                subclass_labels=os.path.join(config['cluster_dir'], 'clusters.pt')
                if os.path.isdir(config['cluster_dir']) else config['cluster_dir'])

    # Train a model with ERM
    if activ_done and not (config['classification_config']['eval_only']
                           or config['classification_config']['save_act_only']):
        erm_dir = config['activations_dir']
    else:
        if config['classification_config']['eval_only'] or config['classification_config'][
                'save_act_only']:
            erm_dir = config['activations_dir']
            model_path = os.path.join(erm_dir,
                                      f'{config["classification_config"]["eval_mode"]}_model.pt')
            print(f'Loading model from {model_path}...')
            model.load_state_dict(torch.load(model_path)['state_dict'])
        erm_dir = harness.classify(config['classification_config'], model, dataloaders,
                                   mode=first_mode)

    if config['classification_config']['eval_only'] or config['classification_config'][
            'save_act_only']:
        exit()

    if config['mode'] == 'george':
        if not config['classification_config']['bit_pretrained'] and not rep_done:
            model.load_state_dict(torch.load(os.path.join(erm_dir, 'best_model.pt'))['state_dict'])

        set_seed(config['seed'], use_cuda)
        # Dimensionality-reduce the model activations
        if rep_done:
            reduction_dir = config['representation_dir']
        else:
            reduction_model = harness.get_reduction_model(config, nn_model=model)
            reduction_dir = harness.reduce(config['reduction_config'], reduction_model,
                                           inputs_path=os.path.join(erm_dir, 'outputs.pt'))

        # Cluster the per-superclass features
        if cluster_done:
            cluster_dir = config['cluster_dir']
        else:
            cluster_model = harness.get_cluster_model(config)
            cluster_dir = harness.cluster(config['cluster_config'], cluster_model,
                                          inputs_path=os.path.join(reduction_dir, 'outputs.pt'))

        set_seed(config['seed'], use_cuda)  # reset random state
        dataloaders = harness.get_dataloaders(
            config, mode='george', subclass_labels=os.path.join(cluster_dir, 'clusters.pt'))
        model = harness.get_nn_model(config, num_classes=num_classes, mode='george')

        # Train the final (GEORGE) model
        george_dir = harness.classify(config['classification_config'], model, dataloaders,
                                      mode='george')


if __name__ == '__main__':
    main()
