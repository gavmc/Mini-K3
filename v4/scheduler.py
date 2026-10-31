import math

from torch.optim.lr_scheduler import _LRScheduler

from config import SchedulerConfig


class WSDScheduler(_LRScheduler):
    def __init__(self, optimizer, cfg: SchedulerConfig, last_epoch=-1):
        self.base_lr = cfg.base_lr
        self.warmup = cfg.warmup
        self.cooldown = cfg.cooldown
        self.total = cfg.total

        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch >= self.total:
            lr = 0.0

        elif self.last_epoch < self.warmup:
            lr = self.base_lr * (self.last_epoch + 1) / self.warmup
        
        elif self.cooldown > 0 and self.last_epoch >= self.total - self.cooldown:
            progress = (self.last_epoch - (self.total - self.cooldown)) / self.cooldown
            lr = self.base_lr * (1.0 - math.sqrt(progress))
        
        else:
            lr = self.base_lr

        return [lr for _ in self.optimizer.param_groups]
