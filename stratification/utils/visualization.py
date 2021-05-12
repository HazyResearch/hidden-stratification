from collections import defaultdict
import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skl
import torch


def visualize_clusters_by_group(
    activations,
    cluster_assignments,
    group_assignments,
    true_subclass_labels=None,
    group_to_k=None,
    save_dir=None,
):
    """
    group_to_k (optional) allows standardization across splits, otherwise it will just use len(df['cluster'].unique())
    """
    data = {
        'x1': activations[:, 0],
        'x2': activations[:, 1] if activations.shape[1] >= 2 else activations[:, 0],
        'cluster': cluster_assignments,
        'group': group_assignments,
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
            g = sns.scatterplot(
                data=group_df,
                x='x1',
                y='x2',
                hue=plot_type,
                hue_order=cluster_types,
                palette=sns.color_palette('hls', n_colors=n_colors),
                alpha=0.5,
            )
            plot_title = 'Clusters' if plot_type == 'cluster' else 'True subclasses'
            plt.title(f'Superclass {group}: {plot_title}')
            plt.xlabel('')
            plt.ylabel('')
            g.get_figure().savefig(
                os.path.join(save_dir, f'group_{group}_{plot_type}_viz.png'), dpi=300
            )
            g.get_figure().clf()
