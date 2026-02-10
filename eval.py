import os
import numpy as np
import torch
from utils.utils import reformat_eval_inputs, eval_metric


def evaluate(args, model, val_loader, epoch, logger):
    model.eval()
    preds, labels, losses, masks = [], [], [], []

    with torch.no_grad():
        for idx in range(len(val_loader)):
            data = val_loader[idx]
            outputs = model(data)
            loss = model.get_loss(outputs) 
            total_loss = loss['task_loss']
            for k in loss:
                if k != 'task_loss':
                    # print(f"[Val] {k}: {loss[k]}")
                    total_loss += loss[k]
            
            label, logit, umask = data['target'], outputs[0], data['umask']
            label, logit, umask = reformat_eval_inputs(args, label, logit, umask)

            preds.append(logit)
            labels.append(label)
            masks.append(umask)

            losses.append(total_loss.item()*masks[-1].sum())

        # calculate metrics
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)

        val_acc, val_ua, val_f1, val_mae, val_loss = eval_metric(args, labels, preds, masks, "Val", losses, epoch, logger) #acc, ua/corr, mae, loss

    return val_acc, val_ua, val_f1, val_mae, val_loss


def test(args, model, test_loader, is_save=False, phase='test'):
    model.eval()
    preds, labels, masks = [], [], []

    with torch.no_grad():
        for idx in range(len(test_loader)):
            data = test_loader[idx]
            outputs = model(data)
            
            label, logit, umask = data['target'], outputs[0], data['umask']
            label, logit, umask = reformat_eval_inputs(args, label, logit, umask)

            preds.append(logit)
            labels.append(label)
            masks.append(umask)
    
        # calculate metrics
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)

        test_acc, test_ua, test_f1, test_mae, _ = eval_metric(args, labels, preds, masks, "Test") # acc, corr, mae, loss

        if is_save:
            # save test whole results
            if len(args.modal_MR) > 1:
                save_dir = os.path.join(args.log_path, args.project_name, args.base_model, f"imr_a{args.modal_MR[0]}_v{args.modal_MR[1]}_l{args.modal_MR[2]}") # ./logs/mmin/imr_a0.5_v0.5_l0.5
            else:
                save_dir = os.path.join(args.log_path, args.project_name, args.base_model, f"mr_{args.modal_MR[0]}") # ./logs/mmin/mr_0.5
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            np.save(os.path.join(save_dir, f'{phase}_pred.npy'), preds)  
            np.save(os.path.join(save_dir, f'{phase}_label.npy'), labels) 

            metrics = {
                "accuracy": test_acc,
                "ua": test_ua,
                "f1": test_f1,
                "mae": test_mae,
            }
            np.save(os.path.join(save_dir, f'{phase}_metrics.npy'), metrics)

    return test_acc, test_ua, test_f1, test_mae