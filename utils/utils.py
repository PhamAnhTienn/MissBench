import os
import random
import numpy as np
import torch
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, recall_score, f1_score


def get_optimizer(args, model):
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not recognized.")
    
    return optimizer


def get_scheduler(args, optimizer):
    if args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.niter, eta_min=0)
    else:
        return NotImplementedError(f'learning rate policy [{args.lr_policy}] is not implemented')
    
    return scheduler


def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use deterministic algorithm


def reformat_eval_inputs(args, label, logit, umask):
    if args.dataset_mode == 'con':
        mask = umask.bool()
    else:
        mask = umask.bool().squeeze()
    
    if label.shape[0] == umask.shape[0]:
        label = label[mask]  # [num_valid, 1]
    if logit.shape[0] == umask.shape[0]:
        logit = logit[mask]  # [num_valid, n_classes] or [num_valid, 1]
    umask = umask[mask]      # [num_valid]
    
    label = label.view(-1, 1).detach().cpu().numpy()  # [batch*seqlen, 1]
    umask = umask.view(-1).detach().cpu().numpy()  # [batch*seqlen]
    logit = logit.view(-1, logit.shape[-1]).detach().cpu().numpy()  # [batch*seqlen, n_classes] or [batch*seqlen, 1]

    if args.regression:
        non_zeros = np.array([i for i, e in enumerate(label) if e != 0]) # remove 0 labels
        label = label[non_zeros]
        logit = logit[non_zeros]
        umask = umask[non_zeros]
            
    # if label/logit.shape[0] != umask.shape[0]: already filtered so no need to filter again
    # retun label and logit as [num_valid, ?], i.e. remove all padded positions
    return label, logit, umask


def eval_metric(args, labels, preds, masks, phase, losses=[], epoch=0, logger=None):
    if not args.regression:
        preds = np.argmax(preds, -1) # preds is not applied (log)softmax, but since argmax still works the same, no need to apply (log)softmax
        loss = round(np.sum(losses)/np.sum(masks), 4)
        acc = accuracy_score(labels, preds)  # Weighted Accuracy
        ua = recall_score(labels, preds, average='macro', zero_division=0) # Unweighted Accuracy
        f1 = f1_score(labels, preds, average='weighted', zero_division=0) # Weighted F1 score
        
        if logger is not None:
            logger.info(f"[{phase}][Epoch {epoch}] Loss: {loss:.4f}, WA: {acc:.4f}, UA: {ua:.4f}, F1: {f1:.4f}")
            
        return acc, ua, f1, 0, loss
    
    else:
        loss = round(np.sum(losses)/np.sum(masks), 4)
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        
        acc = accuracy_score((labels > 0), (preds > 0))
        ua = recall_score((labels > 0), (preds > 0), average='macro')
        mae = np.mean(np.absolute(labels - preds.squeeze()))
        corr = np.corrcoef(labels, preds.squeeze())[0][1]
        f1 = f1_score((labels > 0), (preds > 0), average='weighted')

        if logger is not None:
            logger.info(f"[{phase}][Epoch {epoch}] Loss: {loss:.4f}, WA: {acc:.4f}, UA: {ua:.4f}, F1: {f1:.4f}, Corr: {corr:.4f}, MAE: {mae:.4f}")
            
        return acc, corr, f1, mae, loss

    return -1


def save_model(args, model):
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.base_model}_{args.dataset}.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if hasattr(model, 'base_model'):
        state_dict = model.base_model.state_dict()
    else:
        state_dict = model.state_dict()
        
    torch.save({"args": args, "state_dict": state_dict}, checkpoint_path)