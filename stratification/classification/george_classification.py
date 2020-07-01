import os
import logging
from functools import partial

import sklearn.metrics as skl
import numpy as np
import torch
import torch.optim as optimizers
import torch.optim.lr_scheduler as schedulers
import torch.nn.functional as F
from progress.bar import IncrementalBar as ProgressBar

from stratification.classification.utils import AverageMeter, compute_accuracy
from stratification.classification.datasets import LABEL_TYPES, GEORGEDataset
from stratification.classification.losses import init_criterion

from stratification.utils.logger import init_logger, init_epoch_logger
from stratification.utils.utils import format_timedelta, move_to_device, get_learning_rate, concatenate_iterable


PROGRESS_BAR_SUFFIX = '({batch}/{size}) Time {total:} | ETA {eta:} | ' \
                      'Loss: {loss:.3f} | R Loss: {subclass_rob_loss:.3f} | ' \
                      'Acc: {acc:.3f} | R acc: {subclass_rob_acc:.3f} | ' \
                      'TR acc: {true_subclass_rob_acc:.3f}'


def init_optimizer(optimizer_config, model):
    """Initializes the optimizer."""
    optimizer_class = getattr(optimizers, optimizer_config['class_name'])
    return optimizer_class(model.parameters(), **optimizer_config['class_args'])


def init_scheduler(scheduler_config, optimizer):
    """Initializes the learning rate scheduler."""
    scheduler_class = getattr(schedulers, scheduler_config['class_name'])
    return scheduler_class(optimizer, **scheduler_config['class_args'])


def load_state_dicts(load_path, model, optimizer, scheduler, logger):
    """Loads state from a given load path."""
    state = {
        'epoch': 0,
        'best_score': np.nan,
    }
    if load_path != None:
        logger.info(f'Loading state_dict from {load_path}...')
        checkpoint = torch.load(os.path.join(load_path))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    return state


def register_save_activations_hook(model, model_activation_layer, activations_list):
    """Registers a forward pass hook that saves activations.
    
    Args:
        model(nn.Module): A PyTorch model.
        model_activation_layer(str): The name of the module in the network that
            produces the activations of interest.
        activations_list(List[torch.Tensor]) The list in which we should store the
            model activations.
    """
    def save_activations(model, inp, out):
        activations_list.append(out.view(out.size(0), -1))

    for name, m in model.named_modules():
        if name == model_activation_layer:
            return m.register_forward_hook(save_activations)
    return None


class GEORGEClassification:
    """Executes the classification stage of the GEORGE algorithm.

    Args:
        classification_config(dict): Contains the parameters required to execute this step.
            See utils.schema for type information and examples.
        save_dir(str, optional): Directory at which to save logging information.
            If None, logging information is not saved. Default is None.
        use_cuda(bool, optional): If True, enables GPU usage. Default is False.
    """
    def __init__(self, classification_config, save_dir=None, use_cuda=False, log_format='full',
                 has_estimated_subclasses=False):
        self.config = classification_config
        self.save_dir = save_dir
        if self.save_dir:
            self.logger = init_logger('harness.classification', self.save_dir,
                                      log_format=log_format)
            self.epoch_logger = init_epoch_logger(self.save_dir)
            self.logger.info(f'Saving checkpoints to {self.save_dir}...')
        else:
            # initialize logger without FileHandler
            self.logger = logging.getLogger()

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.device = torch.cuda.current_device()

        # uninitialized unless self.train or self.evaluate is called
        self.criterion = None

        # uninitialized unless self.train is called
        self.optimizer = None
        self.scheduler = None

        self.has_estimated_subclasses = has_estimated_subclasses

    def train(self, model, train_dataloader, val_dataloader, robust=False):
        """Trains the given model.
        
        Note:
            Artifacts are only saved if self.save_dir is intialized. Additionally,
            this function assumes that the "step" unit of the scheduler is epoch-based.
            The model is modified in-place, but the model is also returned to match the
            GEORGECluster API.

        Args:
            model(nn.Module): A PyTorch model.
            train_dataloader(DataLoader): The training dataloader. The dataset within must
                subclass GEORGEDataset.
            val_dataloader(DataLoader): The validation dataloader. The dataset within must
                subclass GEORGEDataset.
            robust(bool, optional): Whether or not to apply robust optimization. Affects
                criterion initialization.
                
        Returns:
            model(nn.Module): The best model found during training.
        """
        if self.criterion is None:
            self.criterion = init_criterion(self.config['criterion_config'], robust,
                                            train_dataloader.dataset, self.use_cuda)
        self.optimizer = init_optimizer(self.config['optimizer_config'], model)
        self.scheduler = init_scheduler(self.config['scheduler_config'], self.optimizer)

        # in order to resume model training, load_path must be set explicitly
        load_path = self.config.get('load_path', None)
        self.state = load_state_dicts(load_path, model, self.optimizer, self.scheduler, self.logger)

        num_epochs = self.config['num_epochs']
        checkpoint_metric = self.config['checkpoint_metric']

        self.logger.basic_info('Starting training.')
        for epoch in range(num_epochs):
            self.state['epoch'] = epoch
            self.scheduler.last_epoch = epoch - 1

            curr_lr = get_learning_rate(self.optimizer)
            self.logger.basic_info(f'\nEpoch: [{epoch + 1} | {num_epochs}] LR: {curr_lr:.2E}')

            self.logger.basic_info('Training:')
            train_metrics, _ = self._run_epoch(model, train_dataloader, optimize=True,
                                               save_activations=False)
            self.logger.basic_info('Validation:')
            val_metrics, _ = self._run_epoch(model, val_dataloader, optimize=False,
                                             save_activations=False)
            metrics = {
                **{f'train_{k}': v
                   for k, v in train_metrics.items()},
                **{f'val_{k}': v
                   for k, v in val_metrics.items()}
            }
            self._checkpoint(model, metrics, checkpoint_metric, epoch)
            self.epoch_logger.append({'learning_rate': curr_lr, **metrics})

            self.scheduler.step(*([self.state[f'best_score']] if type(self.scheduler) ==
                                  schedulers.ReduceLROnPlateau else []))
        torch.cuda.empty_cache()

        best_model_path = os.path.join(self.save_dir, 'best_model.pth.tar')
        if os.path.exists(best_model_path):
            self.logger.basic_info('\nTraining complete. Loading best model.')
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            self.logger.basic_info('Training complete. No best model found.')
        return model

    def evaluate(self, model, dataloader, robust=False, save_activations=False):
        """Evaluates the model.
        
        Note:
            The latter item in the returned tuple is what is necessary to run 
            GEORGECluster.train and GEORGECluster.evaluate.
        
        Args:
            model(nn.Module): A PyTorch model.
            dataloader(DataLoader): The dataloader. The dataset within must
                subclass GEORGEDataset.
            robust(bool, optional): Whether or not to apply robust optimization. Affects
                criterion initialization.
            save_activations(bool, optional): If True, saves the activations in
                `outputs`. Default is False.
                
        Returns:
            metrics(Dict[str, Any]) A dictionary object that stores the metrics defined
                in self.config['metric_types'].
            outputs(Dict[str, Any]) A dictionary object that stores artifacts necessary
                for model analysis, including labels, activations, and predictions.
        """
        # use criterion from training if trained; else, load a new one
        if self.criterion is None:
            self.criterion = init_criterion(self.config['criterion_config'], robust,
                                            dataloader.dataset, self.use_cuda)

        metrics, outputs = self._run_epoch(model, dataloader, optimize=False,
                                           save_activations=save_activations)
        return metrics, outputs

    def _run_epoch(self, model, dataloader, optimize=False, save_activations=False):
        """Runs the model on a given dataloader.
        
        Note:
            The latter item in the returned tuple is what is necessary to run 
            GEORGECluster.train and GEORGECluster.evaluate.
        
        Args:
            model(nn.Module): A PyTorch model.
            dataloader(DataLoader): The dataloader. The dataset within must
                subclass GEORGEDataset.
            optimize(bool, optional): If True, the model is trained on self.criterion.
            save_activations(bool, optional): If True, saves the activations in
                `outputs`. Default is False.
                
        Returns:
            metrics(Dict[str, Any]) A dictionary object that stores the metrics defined
                in self.config['metric_types'].
            outputs(Dict[str, Any]) A dictionary object that stores artifacts necessary
                for model analysis, including labels, activations, and predictions.
        """
        dataset = dataloader.dataset
        self._check_dataset(dataset)
        type_to_num_classes = {
            label_type: dataset.get_num_classes(label_type)
            for label_type in LABEL_TYPES if label_type in dataset.Y_dict.keys()
        }
        outputs = {
            'metrics': None,
            'activations': [],
            'superclass': [],
            'subclass': [],
            'true_subclass': [],
            'targets': [],
            'probs': [],
            'preds': [],
            'losses': [],
        }
        activations_handle = self._init_activations_hook(model, outputs['activations'])
        if optimize:
            progress_prefix = 'Training'
            model.train()
        else:
            progress_prefix = 'Evaluation'
            model.eval()

        per_class_meters = self._init_per_class_meters(type_to_num_classes)
        metric_meters = {k: AverageMeter() for k in ['loss', 'acc']}

        progress = self.config['show_progress']
        if progress:
            bar = ProgressBar(progress_prefix, max=len(dataloader), width=50)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_size = len(targets)
            if self.use_cuda:
                inputs, targets = move_to_device([inputs, targets], device=self.device)

            type_to_labels = {}
            for label_type in type_to_num_classes.keys():
                type_to_labels[label_type] = targets[label_type]
                outputs[label_type].append(targets[label_type])

            if optimize:
                logits = model(inputs)
                loss_targets = targets['superclass']
                co = self.criterion(logits, loss_targets, targets['subclass'])
                loss, (losses, corrects), _ = co
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    logits = model(inputs)
                    loss_targets = targets['superclass']
                    co = self.criterion(logits, loss_targets, targets['subclass'])
                    loss, (losses, corrects), _ = co

            if not save_activations:
                outputs['activations'].pop()  # delete activations

            metrics = self._compute_progress_metrics(losses, corrects, type_to_labels,
                                                     type_to_num_classes, per_class_meters)
            acc, preds = compute_accuracy(logits.data, loss_targets.data, topk=(1, ),
                                          return_preds=True)
            acc = acc[0]
            metric_meters['loss'].update(loss, batch_size)
            metric_meters['acc'].update(acc, batch_size)

            outputs['probs'].append(F.softmax(logits, dim=1).detach().cpu())
            outputs['preds'].append(preds)
            outputs['losses'].append(losses.detach().cpu())
            outputs['targets'].append(loss_targets.detach().cpu())

            if progress:
                bar.suffix = PROGRESS_BAR_SUFFIX.format(
                    batch=batch_idx + 1, size=len(dataloader),
                    total=format_timedelta(bar.elapsed_td), eta=format_timedelta(bar.eta_td), **{
                        **metrics,
                        **{k: v.avg
                           for k, v in metric_meters.items()}
                    })
                bar.next()
        if progress:
            bar.finish()
        if activations_handle:
            activations_handle.remove()

        for k, v in outputs.items():
            if type(v) == list and len(v) > 0:
                outputs[k] = concatenate_iterable(v)
        outputs['metrics'] = metrics
        outputs['metrics'].update({k: float(v.avg) for k, v in metric_meters.items()})
        outputs['metrics'].update(self._compute_aggregate_metrics(outputs))
        self.logger.info(outputs['metrics'])
        if self.has_estimated_subclasses:
            self.logger.basic_info(
                f'Loss: {outputs["metrics"]["loss"]:.3f}, '
                f'Acc.: {outputs["metrics"]["acc"]:.2f}%, '
                f'Est. rob. loss: {outputs["metrics"]["subclass_rob_loss"]:.3f}, '
                f'Est. rob. acc: {outputs["metrics"]["subclass_rob_acc"]:.2f}%, '
                f'True rob. loss: {outputs["metrics"]["true_subclass_rob_loss"]:.3f}, '
                f'True rob. acc: {outputs["metrics"]["true_subclass_rob_acc"]:.2f}%')
        else:
            # "Robust accuracy" is accuracy on the estimated subclasses. If there are none (i.e., we either have
            # no estimate of the subclass labels, or we know the true subclasses), then it is meaningless.
            self.logger.basic_info(
                f'Loss: {outputs["metrics"]["loss"]:.3f}, '
                f'Acc.: {outputs["metrics"]["acc"]:.2f}%, '
                f'True robust loss: {outputs["metrics"]["true_subclass_rob_loss"]:.3f}, '
                f'True robust acc.: {outputs["metrics"]["true_subclass_rob_acc"]:.2f}%')

        return outputs['metrics'], outputs

    def _check_dataset(self, dataset):
        """Checks the validity of the dataset."""
        assert isinstance(dataset, GEORGEDataset), 'Dataset must subclass GEORGEDataset.'
        assert 'subclass' in dataset.Y_dict.keys()

    def _init_activations_hook(self, model, activations_list):
        """Initializes the forward hook to save model activations."""
        if isinstance(model, torch.nn.DataParallel):
            activation_layer = model.module.activation_layer_name
        else:
            activation_layer = model.activation_layer_name
        activations_handle = register_save_activations_hook(model, activation_layer,
                                                            activations_list)
        if activation_layer is not None:
            assert activations_handle is not None, \
                f'No hook registered for activation_layer={activation_layer}'
        return activations_handle

    def _init_per_class_meters(self, type_to_num_classes):
        """Initializes per_class_meters for loss and accuracy.
        
        Args:
            type_to_num_classes(Dict[str, int]): Dictionary object that maps the 
                label_type (e.g. superclass, subclass, true_subclass) to the number
                of classes for that label_type.
            
        Returns:
            per_class_meters(Dict[str, List[AverageMeter]]): A dictionary of
                per_class_meters, where a per_class_meter is a list of AverageMeter 
                objects, one for each class. There is a per_class_meter for each 
                label_type, and for each metric_type (e.g. losses, accs). The 
                AverageMeter objects are used to track metrics on individual groups.
        """
        per_class_meters = {}
        for label_type, num_classes in type_to_num_classes.items():
            for metric_type in ['losses', 'accs']:
                per_class_meter_name = f'per_{label_type}_{metric_type}'
                per_class_meter = [AverageMeter() for i in range(num_classes)]
                per_class_meters[per_class_meter_name] = per_class_meter
        return per_class_meters

    def _compute_progress_metrics(self, sample_losses, corrects, type_to_labels,
                                  type_to_num_classes, per_class_meters):
        """Extracts metrics from each of the per_class_meters.
        
        Args:
            sample_losses(np.ndarray of shape (N, )): The loss computed for
                each sample.
            corrects(np.ndarray of shape(N, )): Whether or not the model produced
                a correct prediction for each sample.
            type_to_labels(Dict[str, Union[np.ndarray, torch.Tensor, Sequence]]):
                Dictionary object mapping the label_type (e.g. superclass, subclass,
                true_subclass) to the labels themselves.
            type_to_num_classes(Dict[str, int]):
                type_to_num_classes(Dict[str, int]): Dictionary object that maps the 
                label_type to the number
                of classes for that label_type.
            per_class_meters(Dict[str, List[AverageMeter]]):  A dictionary of
                per_class_meters, where a per_class_meter is a list of AverageMeter 
                objects, one for each class. There is a per_class_meter for each 
                label_type, and for each metric_type (e.g. losses, accs).
            
        Returns:
            metrics(Dict[str, Any]): A dictionary object that describes model
                performance based on information in each of the per_class_meters.
        """
        batch_stats = {}
        for label_type, labels in type_to_labels.items():
            num_classes = type_to_num_classes[label_type]
            losses, counts = self.criterion.compute_group_avg(sample_losses, labels,
                                                              num_groups=num_classes)
            accs, _ = self.criterion.compute_group_avg(corrects, labels, num_groups=num_classes)
            batch_stats[label_type] = {'losses': losses, 'counts': counts, 'accs': accs}
        metrics = {}
        for label_type, stats in batch_stats.items():
            losses, counts, accs = stats['losses'], stats['counts'], stats['accs']
            loss_meters = per_class_meters[f'per_{label_type}_losses']
            acc_meters = per_class_meters[f'per_{label_type}_accs']

            num_classes = type_to_num_classes[label_type]
            for i in range(num_classes):
                loss_meters[i].update(losses[i], counts[i])
                acc_meters[i].update(accs[i], counts[i])

            active = np.array([i for i, m in enumerate(acc_meters) if m.count])
            if len(active) > 0:
                rob_loss = max([gl.avg for gl in np.array(loss_meters)[active]])
                rob_acc = min([ga.avg * 100 for ga in np.array(acc_meters)[active]])
            else:
                rob_loss = 0.0
                rob_acc = 0.0
            metrics[f'{label_type}_rob_loss'] = rob_loss
            metrics[f'{label_type}_rob_acc'] = rob_acc

        if 'true_subclass_rob_acc' not in metrics.keys():
            metrics['true_subclass_rob_acc'] = -1
        return metrics

    def _compute_aggregate_metrics(self, outputs, evaluate=False):
        """Extracts metrics from the outputs object."""
        return {}

    def _checkpoint(self, model, metrics, checkpoint_metric, epoch):
        """Saves the model.
        
        Args:
            model(nn.Module): A PyTorch model.
            metrics(Dict[str, Any]): A dictionary object containing
                model performance metrics.
            checkpoint_metric(str): The checkpoint metric associated with the model.
            epoch(int): The current epoch. 
        """
        if checkpoint_metric not in metrics.keys():
            raise KeyError(f'{checkpoint_metric} not in metrics {metrics.keys()}')

        if np.isnan(self.state['best_score']):
            self.state['best_score'] = metrics[checkpoint_metric]
            is_best = True
        else:
            if 'loss' in checkpoint_metric:
                is_best = self.state['best_score'] > metrics[checkpoint_metric]
                self.state['best_score'] = min(self.state['best_score'], metrics[checkpoint_metric])
            else:
                is_best = self.state['best_score'] < metrics[checkpoint_metric]
                self.state['best_score'] = max(self.state['best_score'], metrics[checkpoint_metric])

        data = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_score': self.state['best_score'],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            **metrics
        }

        if is_best:
            self._save(data, f'best_model.pth.tar')

        if self.config['save_every'] > 0 and epoch % self.config['save_every'] == 0:
            self._save(data, f'checkpoint_epoch_{epoch}.pth.tar')

    def _save(self, data, filename):
        """If self.save_dir is not None, saves `data`."""
        if self.save_dir is not None:
            filepath = os.path.join(self.save_dir, filename)
            self.logger.info(f'Saving checkpoint to {filepath}...')
            torch.save(data, filepath)
        else:
            self.logger.info('save_dir not initialized. Skipping save step.')
