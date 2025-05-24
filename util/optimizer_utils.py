from torch.optim import lr_scheduler
import numpy as np

class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, *, warm_up=1, T_max=10, start_ratio=0.0):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0    # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:            
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 *\
                            (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)
            
        self.cur += 1
        return [lr for base_lr in self.base_lrs]
