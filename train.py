import copy
import os
import time
import numpy as np

from eval import evaluate, test
from data import create_dataset
from config.config_manager import ConfigManager
from models import Model

from utils.logger import get_console_logger, get_comet_logger
from utils.utils import get_optimizer, get_scheduler, eval_metric, save_model, set_seed, reformat_eval_inputs

from eval_balance import MLI, MEI

if __name__ == '__main__':
    config_manager = ConfigManager()
    args = config_manager.get_args()
    set_seed(args.seed)
    
    # Logging and result directory
    result_dir = os.path.join(args.log_path, args.project_name, args.base_model)  # /logs/mmin/results
    os.makedirs(result_dir, exist_ok=True)

    logger = get_console_logger()
    comet_logger = get_comet_logger("test", args)
    device = args.device
    
    # Get data loader
    dataloaders, embedding_dim = create_dataset(args)
    args.embedding_dim = embedding_dim
    train_loader, val_loader, test_loader = dataloaders
    
    train_size, val_size, test_size = len(train_loader.data), len(val_loader.data), len(test_loader.data)

    logger.info(f'The number of training samples: {train_size}')
    logger.info(f'The number of validation samples: {val_size}')
    logger.info(f'The number of testing samples: {test_size}')

    # Build model
    model = Model(args)
    model.to(device)
    # Get optimizer and scheduler
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    # MEI evaluator
    mei = MEI(args) if args.MEI==True else None
    
    # MLI evaluator
    mli = MLI(args) if args.MLI==True else None

    # Tranining
    best_val_epoch = -1
    best_val_metrics = {
        'wa': 0.0,
        'ua': 0.0,
        'f1': 0.0,
        'mae': 1e8,
        'loss': 1e8
    }
    select_metric = args.select_metric
    best_state = None
    iter = 0
    early_stopping_count = 0

    for epoch in range(args.epochs):
        model.train()
        
        start = time.time()

        preds, labels, losses, masks = [], [], [], []

        for idx in range(len(train_loader)):
            model.zero_grad()
            data = train_loader[idx]
            
            outputs = model(data) # [logits, ?]
            loss = model.get_loss(outputs)  # loss
            total_loss = loss['task_loss']
            
            for k in loss:
                if k != 'task_loss':
                    total_loss += loss[k]
                    
            if mli is not None:
                mli.step_evaluate(model, outputs, data, iter, logger, comet_logger)
            total_loss.backward()
                
            model.clip_gradients(args.max_grad_norm)
                
            if args.modulation and args.start_mod <= epoch <= args.end_mod:
                model.apply_modulation(outputs)
        
            optimizer.step()
            
            label, logit, umask = data['target'], outputs[0], data['umask']
            label, logit, umask = reformat_eval_inputs(args, label, logit, umask)

            preds.append(logit)
            labels.append(label)
            masks.append(umask)
            losses.append(total_loss.item()*masks[-1].sum())
            
            if comet_logger is not None:
                comet_logger.log_metric("train_loss", total_loss.item(), step=iter)
            iter += 1

        # calculate metrics
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
        train_acc, train_ua, train_f1, train_mae, train_loss = eval_metric(args, labels, preds, masks, "Train", losses, epoch, logger)
        
        # eval
        val_acc, val_ua, val_f1, val_mae, val_loss = evaluate(args, model, val_loader, epoch, logger)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)

        logger.info(f'Epoch {epoch} | Time: {duration:.4f} sec | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}')
        if comet_logger is not None:
            comet_logger.log_metric("train_loss", train_loss, epoch=epoch)
            comet_logger.log_metric("train_acc", train_acc, epoch=epoch)
            comet_logger.log_metric("train_f1", train_f1, epoch=epoch)
            comet_logger.log_metric("val_loss", val_loss, epoch=epoch)
            comet_logger.log_metric("val_acc", val_acc, epoch=epoch)
            comet_logger.log_metric("val_f1", val_f1, epoch=epoch)

        metrics = {
            'wa': val_acc,
            'ua': val_ua,
            'f1': val_f1,
            'mae': val_mae,
            'loss': val_loss
        }
        
        if val_loss > 0.0:
            if select_metric in ['wa', 'ua', 'f1'] and metrics[select_metric] > best_val_metrics[select_metric] \
            or select_metric in ['mae', 'loss'] and metrics[select_metric] < best_val_metrics[select_metric]:
                best_val_epoch = epoch
                best_val_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
                early_stopping_count = 0
            else:
                early_stopping_count += 1
        else:
            early_stopping_count += 1
            
        # MEI evaluation
        if mei is not None and epoch % args.MEI_epoch == 0:
            mei.evaluate(args, model, val_loader, logger, comet_logger, phase='Val')
            
        if args.early_stop == early_stopping_count:
            logger.info(f'Early stopping at epoch {epoch}')
            break

    logger.info(f'Best val epoch {best_val_epoch} found with {select_metric} {best_val_metrics[select_metric]}')
    
    
    # Test results
    model.load_state_dict(best_state)
    save_model(args, model)

    best_test_acc, best_test_ua, best_test_f1, best_test_mae = test(args, model, test_loader, is_save=True, phase='test')

    if not args.regression:
        logger.info(f"[Test] Acc: {best_test_acc:.4f}, UA: {best_test_ua:.4f}, F1: {best_test_f1:.4f}")
    else:
        logger.info(f"[Test] Acc: {best_test_acc:.4f}, Corr: {best_test_ua:.4f}, F1: {best_test_f1:.4f}, MAE: {best_test_mae:.4f}")

    if comet_logger is not None:
        comet_logger.log_metric("best_test_acc", best_test_acc)
        comet_logger.log_metric("best_test_ua", best_test_ua)
        comet_logger.log_metric("best_test_f1", best_test_f1)
    
    if mli is not None:
        mli.evaluate(logger, comet_logger)
        
    if mei is not None:
        mei.evaluate(args, model, test_loader, logger, comet_logger, phase='Test')