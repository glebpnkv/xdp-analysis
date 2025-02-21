import copy
import operator as op
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from tqdm.auto import tqdm

from ca_tcc.models.loss import NTXentLoss, SupConLoss
from ca_tcc.utils import _logger
from data.mobility_lab import TrainingMode
from models.mobility_lab import MobilityLabConfigs, MobilityLabModel


def train_step(
    model: MobilityLabModel,
    train_loader,
    optimizer,
    device,
    training_mode: TrainingMode,
    supervised_criterion: _Loss,
    temperature: float = 0.15,
    use_cosine_similarity: bool = True,
):
    model.train()
    total_loss = []
    total_acc = []

    # Iterating over train data
    for batch_idx, (data, labels, aug1, aug2) in tqdm(enumerate(train_loader)):
        data, labels = data.to(device), labels.to(device)
        aug1, aug2 = aug1.to(device), aug2.to(device)
        
        # Optimizer
        optimizer.zero_grad()
        
        if training_mode == "self_supervised" or training_mode == "SupCon":
            predictions1, _, features1 = model(aug1)
            predictions2, _, features2 = model(aug2)

            # Normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_feat1 = model.forward_tc(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = model.forward_tc(features2, features1)

        if training_mode == "self_supervised":
            batch_size = data.shape[0]

            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(
                device,
                batch_size,
                temperature,
                use_cosine_similarity
            )
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                   nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2
        elif training_mode == "SupCon":
            lambda1 = 0.01
            lambda2 = 0.1
            sup_contrastive_criterion = SupConLoss(device)

            sup_con_features = torch.cat(
                tensors=[temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)],
                dim=1
            )
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                    sup_contrastive_criterion(sup_con_features, labels) * lambda2
        else:
            _, logits, features = model(data)
            loss = supervised_criterion(logits, labels)
            total_acc.append(labels.eq(logits.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()

    logs = {
        "train_loss": total_loss,
        "train_acc": total_acc,
    }
    return logs


def val_step(
    model: MobilityLabModel,
    loader,
    device,
    training_mode: TrainingMode,
    supervised_criterion: _Loss,
    metric_prefix: str = "val",
):
    model.eval()

    total_loss = []
    total_acc = []

    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in loader:
            data, labels = data.to(device), labels.to(device)

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                _, logits, features = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                loss = supervised_criterion(logits, labels)
                total_acc.append(labels.eq(logits.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = logits.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        outs = []
        trgs = []

    else:
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc

    logs = {
        f"{metric_prefix}_loss": total_loss,
        f"{metric_prefix}_acc": total_acc,
        f"{metric_prefix}_preds": outs,
        f"{metric_prefix}_trgs": trgs,
    }
    return logs

def train_model(
    model: MobilityLabModel,
    config: MobilityLabConfigs,
    training_mode: TrainingMode,
    train_loader,
    optimizer,
    device,
    epochs,
    experiment_log_dir,
    checkpoint_metric: str = "val_acc",
    checkpoint_mode: str = "max",
    val_loader = None,
    test_loader = None
):
    logger = _logger("train-mobility-lab")

    # Start training
    logger.debug("Training started")

    supervised_criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min"
    )

    last_best_metric = np.inf if checkpoint_mode == "min" else -np.inf
    comparison_op = op.lt if checkpoint_mode == "min" else op.gt

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Train step
        logger.debug(f"Training epoch {epoch} started")

        logger.debug(f"Training step of epoch {epoch} started")
        train_log = train_step(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            training_mode=training_mode,
            supervised_criterion=supervised_criterion,
            temperature=config.temperature,
            use_cosine_similarity=config.use_cosine_similarity,
        )
        train_loss = train_log["train_loss"]
        train_acc = train_log["train_acc"]

        log = train_log

        if val_loader:
            # Validation step
            logger.debug(f"Validation step of epoch {epoch} started")
            val_log = val_step(
                model=model,
                loader=val_loader,
                device=device,
                training_mode=training_mode,
                supervised_criterion=supervised_criterion,
            )
            val_loss = val_log["val_loss"]
            val_acc = val_log["val_acc"]

            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                scheduler.step(val_loss)

            logger.debug(f'\nEpoch : {epoch}\n'
                         f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                         f'Validation Loss     : {val_loss:2.4f}\t | \tValidation Accuracy     : {val_acc:2.4f}')

            log = log | val_log
        if checkpoint_metric in log:
            metric = log[checkpoint_metric]
            if comparison_op(metric, last_best_metric):
                last_best_metric = metric
                chkpoint = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                } | log
                torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", "ckp_best.pt"))
                logger.debug(f"Checkpoint saved at epoch {epoch} with {checkpoint_metric} = {metric}")

    if test_loader and (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # Evaluate on the test set
        logger.debug('Best saved model on the Test set:')

        best_model = copy.deepcopy(model)
        best_model_path = os.path.join(experiment_log_dir, "saved_models", "ckp_best.pt")
        best_model_dict = torch.load(
            best_model_path,
            weights_only=False
        )
        model_dict = best_model.state_dict()
        model_dict.update(best_model_dict["model_state_dict"])
        best_model.load_state_dict(model_dict)

        test_log = val_step(
            model=best_model,
            loader=test_loader,
            device=device,
            training_mode=training_mode,
            supervised_criterion=supervised_criterion,
            metric_prefix="test"
        )
        test_loss, test_acc = test_log["test_loss"], test_log["test_acc"]

        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')
        logger.close()
