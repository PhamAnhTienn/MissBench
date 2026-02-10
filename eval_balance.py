import numpy as np
import torch
import copy
from utils.utils import reformat_eval_inputs, eval_metric
from models.loss import MaskedMSELoss, MaskedCELoss

class MEI:
    def __init__(self):
        self.metrics = ['acc', 'ua', 'f1']
        self.modal = 'AVL'

    def get_modal_eval(self, args, model, org_val_loader, eval_modal):
        preds, labels, masks = [], [], []

        val_loader = copy.deepcopy(org_val_loader)
        with torch.no_grad():
            for idx in range(len(val_loader)):
                data = val_loader[idx]
                for modal in self.modal:
                    if modal not in eval_modal:
                        data[f'{modal}_miss_index'] = torch.zeros_like(
                            data[f'{modal}_miss_index']).long().to(args.device)
                    else:
                        data[f'{modal}_miss_index'] = torch.ones_like(
                            data[f'{modal}_miss_index']).long().to(args.device)

                outputs = model(data)

                label, logit, umask = data['target'], outputs[0], data['umask']
                label, logit, umask = reformat_eval_inputs(args, label, logit, umask)

                preds.append(logit)
                labels.append(label)
                masks.append(umask)


            # calculate metrics
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            masks = np.concatenate(masks)

            val_acc, val_ua, val_f1, val_mae, _ = eval_metric(
                args, labels, preds, masks, "Val")  # acc, ua/corr, f1, 0/mae, loss
            return {
                'acc': val_acc,
                'ua': val_ua,
                'f1': val_f1,
                'mae': val_mae,
            }

    def evaluate(self, args, model, val_loader, logger=None, comet_logger=None, phase='Val'):
        model.eval()
        base_eval = self.get_modal_eval(
            args, model, val_loader, eval_modal='AVL')

        modal_drop_eval = {m: {'acc': [], 'ua': [], 'f1': [],
                    'mae': []} for m in self.modal}
        for eval_modal in ['AV', 'AL', 'VL', 'A', 'V', 'L']:
            eval_res = self.get_modal_eval(
                args, model, val_loader, eval_modal=eval_modal)
            if comet_logger is not None:
                for metric in self.metrics:
                    comet_logger.log_metric(
                        f'[{metric.upper()}]_{phase}_{eval_modal}', eval_res[metric])
            for modal in self.modal:
                for metric in self.metrics:
                    if modal not in eval_modal:
                        modal_drop_eval[modal][metric].append(
                            eval_res[metric])
                        
        if comet_logger is not None:
            for metric in self.metrics:
                comet_logger.log_metric(
                    f'[{metric.upper()}]_{phase}_base', base_eval[metric])
                 
        for metric in self.metrics:   
            norm_contr = [] 
            for modal in modal_drop_eval:
                base = base_eval[metric]
                vals = modal_drop_eval[modal][metric]
                diffs = [base - v for v in vals]
                avg_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                meanvstd = avg_diff / (std_diff+1e-18)
                if comet_logger is not None:
                    comet_logger.log_metric(
                        f'[{metric.upper()}]_{phase} drop_meanvstd_{modal}', meanvstd)
                    comet_logger.log_metric(
                        f'[{metric.upper()}]_{phase} drop_meanvstd_abs_{modal}', np.abs(meanvstd))
                norm_contr.append(np.abs(meanvstd))
                
            prob_contr = [x / (sum(norm_contr)+1e-18) for x in norm_contr]
            entropy = -np.log(sum([p*p for p in prob_contr])+1e-18)
            mei = max(0, 1 -entropy / np.log(len(prob_contr)+1e-18))
            if comet_logger is not None:
                for idx, m in enumerate(self.modal):
                    comet_logger.log_metric(
                        f'[{metric.upper()}]_{phase}_prob_{m}', prob_contr[idx])
                comet_logger.log_metric(
                    f'[{metric.upper()}]_{phase} entropy', entropy)
                comet_logger.log_metric(
                    f'[{metric.upper()}]_{phase} MEI', mei)
                comet_logger.log_metric(
                    f'[{metric.upper()}]_{phase} MEI_root', mei**(1/len(self.modal)))
            
            if logger is not None:
                logger.info(
                    f"[{metric.upper()}] [MEI] {mei:.4f}, mean_v_std A/V/L: {norm_contr}, prob_contr: {prob_contr}")
        
        return mei

class MLI:
    def __init__(self, args):
        self.regression = args.regression
        self.modal = args.modal
        self.iter = 0
        self.log_iter = 100000
        self.modal_fluc = {m: [] for m in self.modal}
        self.prev_grad = {m: None for m in self.modal}
        
        if self.regression:
            self.criterion = MaskedMSELoss(args, reduction='none')
        else:
            self.criterion = MaskedCELoss(args, reduction='none')

    def get_gradient(self, model, loss):
        grads = torch.autograd.grad(
            loss,
            model.parameters(),
            retain_graph=True,
            allow_unused=True
        )
        return grads

    def get_modal_grad(self, grads):
        l2_norm = torch.stack([g.float().norm(p=2)
                              for g in grads if torch.is_tensor(g)])
        mean = torch.mean(l2_norm)
        return mean.item()

    def step_evaluate(self, model, output, inputs, comet_logger=None):
        target = model.target
        umask = model.umask
        with torch.enable_grad():
            all_loss = self.criterion(
                output[0], target, umask).squeeze()  # [num_valid]
        
        for m in self.modal:
            modal_mask = inputs[f'{m}_miss_index']  # [B, L]
            if len(modal_mask.shape) < 2:
                modal_mask = modal_mask.unsqueeze(-1)
            assert modal_mask.shape == umask.shape, f'mm:{modal_mask.shape}, um:{umask.shape}'
            modal_mask = modal_mask[umask.bool()]  # [num_valid]
            modal_loss = torch.sum(all_loss*modal_mask)/torch.sum(modal_mask)
            module_grad = self.get_gradient(model, modal_loss)
            modal_stats = self.get_modal_grad(module_grad)
            if self.prev_grad[m] is not None:
                self.modal_fluc[m].append(np.abs(modal_stats - self.prev_grad[m]))
                if np.isnan(self.modal_fluc[m][-1]):
                    self.modal_fluc[m][-1] = np.abs(self.prev_grad[m])
            self.prev_grad[m] = modal_stats
            
            if comet_logger is not None:
                comet_logger.log_metric(
                    f'[MLI] module_mean{m}', modal_stats, step=self.iter)
                if len(self.modal_fluc[m]) > 0:
                    comet_logger.log_metric(
                        f'[MLI] modal_fluc_{m}', self.modal_fluc[m][-1], step=self.iter)
            
        self.iter += 1

    def evaluate(self, comet_logger=None):
        modal = self.modal
        modal_fluc = self.modal_fluc
        num_modal = len(modal)
        total_step = len(modal_fluc[modal[0]])
        
        base_val, diff = [], []
        
        for step in range(total_step):
            step_fluc = {m: modal_fluc[m][step] for m in modal}                
            cur_val = np.mean([step_fluc[m] for m in modal])
            cur_diff = np.mean([np.abs(cur_val-step_fluc[m]) for m in modal])
            base_val.append(cur_val)
            diff.append(cur_diff)
            
        mli = (np.mean(diff)/np.max(base_val))**(1/num_modal)
        if comet_logger is not None:
            comet_logger.log_metric('[MLI]', mli)
        return mli