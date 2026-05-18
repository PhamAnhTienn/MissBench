import numpy as np
import torch
import copy
from utils.utils import reformat_eval_inputs, eval_metric
from models.loss import MaskedMSELoss, MaskedCELoss
from itertools import combinations

class MEI:
    def __init__(self, args):
        self.metric = args.mei_metric
        self.modal = args.modal
        self.subset = [''.join(c) for r in range(1, len(self.modal)) 
            for c in combinations(self.modal, r)]

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
        metric = self.metric
        perf_full = self.get_modal_eval(
            args, model, val_loader, eval_modal=self.modal)

        perf_c = {m: [] for m in self.modal}
        for eval_modal in self.subset:
            eval_res = self.get_modal_eval(
                args, model, val_loader, eval_modal=eval_modal)
            for modal in self.modal:
                if modal not in eval_modal:
                    perf_c[modal].append(
                        eval_res[metric])
                
        norm_contr = [] 
        for modal in perf_c:
            s_m = [perf_full[metric] - v for v in perf_c[modal]]
            avg_s = np.mean(s_m)
            std_s = np.std(s_m)
            norm_s = avg_s / (std_s+1e-18)
            norm_contr.append(np.abs(norm_s))
            
        prob_contr = [x / (sum(norm_contr)+1e-18) for x in norm_contr]
        entropy = -np.log(sum([p*p for p in prob_contr])+1e-18)
        mei = max(0, 1 -entropy / np.log(len(prob_contr)+1e-18))
        
        if comet_logger is not None:
            comet_logger.log_metric(
                f'[{metric.upper()}]_{phase} MEI', mei)
            for idx, modal in enumerate(self.modal):
                comet_logger.log_metric(
                    f'[{metric.upper()}]_{phase} prob_contr_{modal}', prob_contr[idx])

        if logger is not None:
            logger.info(
                f"[{metric.upper()}] [MEI] {mei:.4f}, prob_contr{self.modal}: {prob_contr}")
        
        return mei

class MLI:
    def __init__(self, args):
        self.regression = args.regression
        self.modal = args.modal
        self.log_iter = args.mli_log_iter
        self.grad_var = {m: [] for m in self.modal}
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

    def step_evaluate(self, model, output, inputs, iter, logger=None, comet_logger=None):
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
            modal_grad = self.get_modal_grad(module_grad)
            if self.prev_grad[m] is not None:
                self.grad_var[m].append(np.abs(modal_grad - self.prev_grad[m]))
                if np.isnan(self.grad_var[m][-1]):
                    self.grad_var[m][-1] = np.abs(self.prev_grad[m])
            self.prev_grad[m] = modal_grad
            
            if comet_logger is not None:
                if len(self.grad_var[m]) > 0:
                    comet_logger.log_metric(
                        f'[MLI] modal_var_{m}', self.grad_var[m][-1], step=iter)
            
            if logger is not None and iter % self.log_iter == 0:
                if len(self.grad_var[m]) > 0:
                    logger.info(
                        f"[MLI] Iter {iter} modal_var_{m}: {self.grad_var[m][-1]:.6f}")
            

    def evaluate(self, logger=None, comet_logger=None):
        modal = self.modal
        grad_var = self.grad_var
        num_modal = len(modal)
        total_step = len(grad_var[modal[0]])
        
        avg_var, diff = [], []
        
        for step in range(total_step):
            step_fluc = {m: grad_var[m][step] for m in modal}                
            cur_var = np.mean([step_fluc[m] for m in modal])
            cur_diff = np.mean([np.abs(cur_var-step_fluc[m]) for m in modal])
            avg_var.append(cur_var)
            diff.append(cur_diff)
            
        mli = (np.mean(diff)/np.max(avg_var))**(1/num_modal)
        if comet_logger is not None:
            comet_logger.log_metric('[MLI]', mli)
        if logger is not None:
            logger.info(f"[MLI]: {mli:.4f}")
        return mli