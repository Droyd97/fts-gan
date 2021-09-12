"""
Loss functions that can be used by a GAN framework
"""
import torch
import torch.nn as nn


class GANLoss(nn.Module):

    def __init__(self, loss_function='standard', topk=None, real_target_label=1.0, fake_target_label=0.0):
        super(GANLoss, self).__init__()
        self.loss_function = loss_function
        self.topk = topk
        if self.loss_function == 'standard':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('loss function {} not implemented'.format(self.loss_function))

    def get_target(self, predicitons, target_is_real):
        if target_is_real:
            target = torch.ones_like(predicitons, device=predicitons.device)
        else:
            target = torch.zeros_like(predicitons, device=predicitons.device)
        return target

    def forward(self, predictions, target_is_real):
        if self.topk is not None:
            predictions = torch.topk(predictions, self.topk, dim=0)[0]
        target = self.get_target(predictions, target_is_real)
        return self.loss(predictions, target)
