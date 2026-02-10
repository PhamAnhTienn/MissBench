import torch.nn as nn
import models.encoder
from models.loss import MaskedCELoss, MaskedMSELoss
import importlib

def find_encoder(args):
    encoder_name = args.base_encoder
    for name, cls in models.encoder.__dict__.items():
        if encoder_name.lower() in name.lower() and issubclass(cls, nn.Module):
            base_encoder = cls
            break
    return base_encoder

def find_basemodel(args):
    basemodel_name = args.base_model
    basemodal_lib = importlib.import_module(f'models.{basemodel_name}.{basemodel_name}_model')

    for name, cls in basemodal_lib.__dict__.items():
        if 'model' in name.lower() and issubclass(cls, nn.Module) and cls.__module__ == basemodal_lib.__name__:
            base_model = cls
            break
    return base_model


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        base_encoder = find_encoder(args)
        base_model = find_basemodel(args)
        
        self.base_model = base_model(args, base_encoder)
        self.regression = args.regression
        if self.regression:
            self.task_loss = MaskedMSELoss(args)
        else:
            self.task_loss = MaskedCELoss(args)
        self.task_weight = args.task_weight
        self.device = args.device
    
    def set_input(self, inputs):
        self.target = inputs['target']  # [batch, seqlen]
        self.umask = inputs['umask']  # [batch, seqlen]
        self.base_model.set_input(inputs)
        
    def forward(self, inputs):
        self.set_input(inputs)
        outputs = self.base_model.get_outputs()
        # outputs: [logits,...] - first element is always logits for the task (cls/reg), remaining are optional outputs of base_model
        return outputs
    
    def get_loss(self, outputs):
        # outputs: [logits,...]
        loss = {
            'task_loss': self.task_weight * self.task_loss(outputs[0], self.target, self.umask)
        }
        
        sub_loss = self.base_model.get_loss(outputs)
        for k in sub_loss:
            loss[k] = sub_loss[k]
            
        return loss
    
    def clip_gradients(self, max_norm: float):
        """
        Clip gradients of the model.
        """
        if max_norm <= 0:
            return

        model_names = getattr(self.base_model, 'model_names', None)

        if model_names:
            # Local gradient clipping
            for name in model_names:
                sub_model = getattr(self.base_model, name, None)
                if sub_model is not None:
                    nn.utils.clip_grad_norm_(
                        sub_model.parameters(),
                        max_norm
                    )
        else:
            # Global gradient clipping
            nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm
            )
        
    def apply_modulation(self, outputs):
        self.base_model.apply_modulation(outputs)