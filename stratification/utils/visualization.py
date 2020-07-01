import os
import random
from collections import defaultdict, Counter
import json
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skl
import pickle
import torch


def visualize_clusters_by_group(activations, cluster_assignments, group_assignments,
                                true_subclass_labels=None, group_to_k=None, save_dir=None):
    """
    group_to_k (optional) allows standardization across splits, otherwise it will just use len(df['cluster'].unique())
    """
    data = {
        'x1': activations[:, 0],
        'x2': activations[:, 1] if activations.shape[1] >= 2 else activations[:, 0],
        'cluster': cluster_assignments,
        'group': group_assignments
    }
    if true_subclass_labels is not None:
        data['true_subclass'] = true_subclass_labels
    df = pd.DataFrame(data)

    groups = np.unique(group_assignments)
    for group in groups:
        group_df = df.loc[df['group'] == group]
        for plot_type in ['cluster', 'true_subclass']:
            if plot_type not in data:
                continue

            cluster_types = sorted(group_df[plot_type].unique())
            if plot_type == 'true_subclass':
                n_colors = len(cluster_types)
            elif plot_type == 'cluster':
                n_colors = group_to_k[group] if group_to_k != None else len(cluster_types)
            g = sns.scatterplot(data=group_df, x='x1', y='x2', hue=plot_type,
                                hue_order=cluster_types,
                                palette=sns.color_palette('hls', n_colors=n_colors), alpha=.5)
            plot_title = 'Clusters' if plot_type == 'cluster' else 'True subclasses'
            plt.title(f'Superclass {group}: {plot_title}')
            plt.xlabel('')
            plt.ylabel('')
            g.get_figure().savefig(os.path.join(save_dir, f'group_{group}_{plot_type}_viz.png'),
                                   dpi=300)
            g.get_figure().clf()


### UNTESTED VISUALIZATIONS ###


def plot(x, y, plot_id, dataset, n_clusters, sub_names=None, sup_name=None, save_dir=None,
         threshold=None):
    """
    Adapted from n2d repo [put in link]
    """
    assert (isinstance(x, np.ndarray) and len(x.shape) == 2)  # and 1 <= x.shape[1] <= 2)
    if x.shape[1] > 2:
        logging.warn(f'Attempting to plot {x.shape[1]} components on a 2D graph!')
        x = x[:, :2]
    if x.shape[1] == 1:
        x = np.hstack((x, np.zeros_like(x)))
    plot_num = min(200000, len(y))
    viz_df = pd.DataFrame(data=x[:plot_num])
    viz_df['Label'] = y[:plot_num]
    if sub_names is not None:
        viz_df['Label'] = viz_df['Label'].map(sub_names)

    if save_dir:
        viz_df.to_csv(save_dir + '/' + dataset + '.csv')
    plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=0, y=1, hue='Label', legend='full',
                    hue_order=sorted(viz_df['Label'].unique()),
                    palette=sns.color_palette('hls', n_colors=n_clusters), alpha=.5, data=viz_df)
    if threshold is not None:
        plt.axvline(threshold, 0, 1, linestyle=':')
    l = plt.legend(bbox_to_anchor=(0, -.14, 1.0, .5), loc='lower left', markerfirst=True,
                   mode='expand', borderaxespad=0, ncol=n_clusters + 1, handletextpad=0.01)

    l.texts[0].set_text('')
    plt.ylabel('')
    plt.xlabel('')

    if sup_name is not None:
        plt.title(f'Superclass: {sup_name}')
    if save_dir:
        plt.savefig(save_dir + '/' + dataset + '-' + plot_id + '.png', dpi=300)
    return plt


def visualize(x, y, y_pred, dataset, n_clusters, n_true_clusters=None, sub_names=None,
              sup_name=None, save_dir=None, threshold=None):
    """
    Adapted from n2d repo [put in link]
    """
    if n_true_clusters is not None:
        plt_y = plot(x, y, 'y', dataset, n_true_clusters, sub_names=sub_names, sup_name=sup_name,
                     save_dir=save_dir, threshold=threshold)

    plt_y_pred = plot(x, y_pred, 'predicted', dataset, n_clusters, sub_names=None,
                      sup_name=sup_name, save_dir=save_dir, threshold=threshold)

    if n_true_clusters is not None and x.shape[-1] == 1:
        plt_hist = plot_histogram(x, y, 'histogram', dataset, n_clusters, sub_names, sup_name,
                                  save_dir, threshold=threshold)
    return plt_y, plt_y_pred


def plot_histogram(x, y, plot_id, dataset, n_subclasses, sub_names, sup_name, save_dir,
                   threshold=None):
    plt.subplots(figsize=(8, 5))

    subclasses = sorted(np.unique(y))
    x_range = (np.amin(x), np.amax(x))
    for i, subclass in enumerate(subclasses):
        sub_data = x[y == subclass]
        plt.hist(sub_data, density=False, bins=50, label=f'{sub_names[i]}', alpha=0.5,
                 range=x_range)
    if threshold is not None:
        plt.axvline(threshold, 0, 1, linestyle=':')
    if sup_name is not None:
        plt.title(f'Superclass: {sup_name}')
    l = plt.legend(bbox_to_anchor=(0, -.14, 1.0, .5), loc='lower left', markerfirst=True,
                   mode='expand', borderaxespad=0, ncol=n_subclasses + 1, handletextpad=0.01)
    if save_dir:
        plt.savefig(save_dir + '/' + dataset + '-' + plot_id + '.png', dpi=300)
    return plt


def plot_and_score_subsets(split, loader, subset_aurocs, golds, probs, save_dir=None,
                           use_fine_true=False):
    subset_idxs_key = 'fine_true' if use_fine_true else 'fine_label'

    plt.close('all')
    for subset_name, subset_idxs in loader.dataset.subset_idxs[subset_idxs_key].items():
        subset_golds = np.array([gold for idx, gold in enumerate(golds) if idx in subset_idxs])
        subset_probs = np.array([prob for idx, prob in enumerate(probs) if idx in subset_idxs])
        subset_auroc = subset_aurocs[subset_name]
        fpr, tpr, _ = skl.roc_curve(subset_golds, subset_probs)

        subset_name = str(subset_name)
        if subset_name[0] == '_':
            subset_name = subset_name[1:]
        plt.plot(fpr, tpr, label=f'{subset_name} ({subset_auroc})')
    plt.legend()
    plt.title(f'Split: {split.capitalize()}')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'subset_roc_curves_{split}.png'))
        plt.close('all')
    return plt


def get_cluster_data(exp_dir, cluster_dir):
    # prevents circular imports
    from eval_representation import get_subclass_preds
    output = {'train': {}, 'val': {}, 'test': {}}

    all_feats = np.load(os.path.join(exp_dir, cluster_dir, f'{cluster_dir}_featurenorm.npz'))
    all_feats = {'train': [all_feats['arr_0'], all_feats['arr_1']], 'val': {}, 'test': {}}

    with open(os.path.join(exp_dir, cluster_dir, 'preds_by_superclass.pkl'), 'rb') as f:
        labels_raw = pickle.load(f)
        pred_sublabels = {}
        for split, supclass_preds in labels_raw.items():
            _, pred_sublabels[split] = get_subclass_preds(supclass_preds)

    # backward compatibility
    val_split_exists = 'val' in pred_sublabels

    saved_acts_list = torch.load(os.path.join(exp_dir, 'saved_activations.pth'))
    if val_split_exists:
        saved_acts = {
            'train': saved_acts_list[0],
            'val': saved_acts_list[1],
            'test': saved_acts_list[2]
        }
        output_json_path = os.path.join(exp_dir, f'test_output.json')
    else:
        saved_acts = {'train': saved_acts_list[0], 'test': saved_acts_list[1]}
        output_json_path = os.path.join(exp_dir, f'output.json')

    with open(output_json_path, 'r') as f:
        gold_output = json.load(f)

    with open(os.path.join(exp_dir, cluster_dir, 'clustering_output.json'), 'r') as f:
        pred_output = json.load(f)

    for split, acts in saved_acts.items():
        suplabels = acts[1]
        y_gold = acts[2]
        y_pred = pred_sublabels[split]
        for superclass, feats in enumerate(all_feats[split]):
            if type(suplabels) == torch.Tensor:
                suplabels = suplabels.cpu().numpy()
            if type(y_gold) == torch.Tensor:
                y_gold = y_gold.cpu().numpy()

            output[split][superclass] = {
                'x': feats,
                'y_gold': y_gold[suplabels == superclass],
                'y_pred': y_pred[suplabels == superclass],
                'metric_y_gold': {
                    str(i): v
                    for i, (
                        _,
                        v) in enumerate(gold_output[str(superclass)]['subclass_accuracies'].items())
                },
                'metric_y_pred': pred_output[str(superclass)][split]['ACC_cluster']
            }
    return output


def visualize_clusters(exp_dir, filename, split, sub_names, sup_names, save_dir=None,
                       save_prefix='default', show_acc=True, fig_size=(20, 5)):
    cluster_data = get_cluster_data(exp_dir, filename)
    cluster_data = cluster_data[split]

    for i, (superclass, coords) in enumerate(cluster_data.items()):
        fig = plt.figure(figsize=fig_size)
        AX = gridspec.GridSpec(1, 2)
        AX.update(wspace=0.08, hspace=0.1)

        superclass_name = f'Superclass: {sup_names[superclass]}'
        subclass_names = sub_names[superclass]
        for j, y_type in enumerate(['y_gold', 'y_pred']):
            x = coords['x']
            y = coords[y_type] - coords[y_type].min()
            n_colors = len(np.unique(y))

            for subclass_idx in subclass_names.keys():
                if show_acc:
                    acc = float(cluster_data[superclass][f'metric_{y_type}'][str(subclass_idx)])
                    subclass_names[subclass_idx] += f" ({acc:.1f})"

            plot_num = min(10000, len(y))
            print(f'plotting {plot_num} points...')
            viz_df = pd.DataFrame(data=x[:plot_num])
            viz_df[superclass_name] = y[:plot_num]
            if y_type == 'y_gold' and subclass_names != None:
                viz_df[superclass_name] = viz_df[superclass_name].map(subclass_names)

            ax = plt.subplot(AX[0, j])
            g = sns.scatterplot(x=0, y=1, hue=superclass_name, legend='full',
                                hue_order=sorted(viz_df[superclass_name].unique()),
                                palette=sns.color_palette('hls',
                                                          n_colors=n_colors), alpha=.5, data=viz_df)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[1:], labels=labels[1:], frameon=True, loc='upper right',
                      bbox_to_anchor=(1.02, 1.05), ncol=1, fancybox=True, shadow=True)

            plt.setp(ax.get_legend().get_texts(), fontsize='24')  # for legend text

            if j == 0:
                ax.set_ylabel(
                    superclass_name,
                    rotation=90,
                    size='24',
                )
            else:
                ax.set_ylabel('')
            ax.set_xlabel('')

            axes = fig.get_axes()
            for ax in axes:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

        plt.tight_layout()
        if save_dir:
            out_name = f'pretty_{save_prefix}_{split}_{y_type}_superclass_{superclass}'
            viz_df.to_csv(os.path.join(save_dir, f'{out_name}.csv'))
            fig.savefig(os.path.join(save_dir, f'{out_name}.pdf'))
            plt.close('all')
    return plt


def visualize_cluster_samples(dataset_name, loader, labels, num_samples=6, save_dir=None,
                              subclass_names='true', alignment='vertical',
                              get_cluster_to_image_names=False, target_clusters=None):
    """
    loader (Dataloader) dataloader for given split
    labels (arraylike) contains the predicted labels for a given split from preds_by_superclass.pkl
    num_samples (int) number of samples to show from each cluster
    save_dir (str, optional) if specified, saves the figure into the given directory
    subclass_names (list or str or None, optional)fl
        if 'true', uses dataset.subclass_names. if list, uses list. if None, uses cluster idx.
    alignment (str) must be either vertical or horizontal. determines the direction that samples go
    """
    assert alignment in {'vertical', 'horizontal'}

    if subclass_names == 'true':
        subclass_names = loader.dataset.true_subclass_names

    transform_fn = loader.dataset.transform
    loader.dataset.transform = None

    cluster_to_idxs = defaultdict(list)
    for idx, cluster in enumerate(labels):
        cluster_to_idxs[int(cluster)].append(idx)
    # preserve cluster order
    cluster_to_idxs = sorted(cluster_to_idxs.items())
    if target_clusters == None:
        target_clusters = cluster_to_idxs
    else:
        target_clusters = [(cluster, idxs) for cluster, idxs in cluster_to_idxs
                           if cluster in target_clusters]

    if dataset_name == 'waterbirds':
        target_clusters = sorted(target_clusters + target_clusters)

    num_clusters = len(target_clusters)
    grid_dims = (num_samples, num_clusters) if alignment == 'vertical' else \
                (num_clusters, num_samples)

    fig = plt.figure(figsize=[3 * d for d in reversed(grid_dims)])
    for i, (cluster, idxs) in enumerate(target_clusters):
        for j, idx in enumerate(random.sample(idxs, num_samples)):
            image = loader.dataset[idx][0]['image']
            coord = (j, i) if alignment == 'vertical' else (i, j)
            plt.subplot2grid(grid_dims, coord)
            plt.imshow(image)

    axes = fig.get_axes()
    for ax in axes:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    for i, (cluster, idxs) in enumerate(target_clusters):
        ax = axes[i * num_samples]
        if subclass_names is not None:
            subclass_name = subclass_names[cluster]
        else:
            subclass_name = cluster

            if dataset_name == 'mnist':
                subclass_name = cluster - 5

        if alignment == 'horizontal':
            ax.set_ylabel(subclass_name, rotation=0, size=48)
            ax.yaxis.labelpad = 40
        if alignment == 'vertical':
            ax.set_xlabel(subclass_name, rotation=0, size=48)
            ax.xaxis.labelpad = 20
            ax.xaxis.set_label_position('top')

    plt.tight_layout()
    if save_dir:
        filename = loader.dataset.__class__.__name__
        if subclass_names:
            filename += f"_{'-'.join(subclass_names)}".replace('/', '-')
        else:
            filename += '_cluster-idxs'
        filename += f'_{num_samples}-samples.pdf'
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path)
        plt.close('all')

    loader.dataset.transform = transform_fn
    if get_cluster_to_image_names:
        cluster_to_image_names = defaultdict(list)
        for cluster, idxs in cluster_to_idxs:
            for idx in idxs:
                image_name = loader.dataset.X_dict['image_name'][idx]
                cluster_to_image_names[cluster].append(image_name)
        return fig, cluster_to_image_names
    return fig


def get_cluster_concentrations(exp_dir, filename, y_true, y_pred, split='test',
                               subclass_names=None):
    with open(os.path.join(exp_dir, filename, 'preds_by_superclass.pkl'), 'rb') as f:
        preds_by_superclass = pickle.load(f)['test']

    y_true_set = set(y_true)

    class_map = {}
    subclass_inc = 0
    for superclass_idx in range(len(preds_by_superclass)):
        subclasses = preds_by_superclass[superclass_idx][0]
        subclasses = sorted(set(subclasses))
        class_map[superclass_idx] = subclasses

    cluster_composition = defaultdict(list)
    for i, cluster_idx in enumerate(y_pred):
        cluster_composition[cluster_idx].append(int(y_true[i]))

    cluster_to_true_counts = {}
    for cluster_idx, cluster_true_labels in cluster_composition.items():
        true_counts = defaultdict(float, Counter(cluster_true_labels))
        cluster_to_true_counts[cluster_idx] = true_counts, sum(true_counts.values())

    if subclass_names == None:
        subclass_names = [i for i in range(y_true_set)]
    data = []
    for cluster_idx, (true_counts, total) in cluster_to_true_counts.items():
        data.append({
            "cluster": cluster_idx,
            **{subclass_names[label]: true_counts[label] / total
               for label in y_true_set}
        })

    concentrations_df = pd.DataFrame(data).set_index('cluster')
    concentrations_df = concentrations_df.sort_values(by='cluster')
    return concentrations_df
