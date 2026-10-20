# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# From MTrans (Liu et al., ECCV 2022; github.com/Cliu2/MTrans) via the team fork github.com/GNaveenkumar614/Transmodel, commit ebf8a90 (2026-10-06), file utils/lr_scheduler.py.
# Modified for OCGF: only WarmupCosineAnnealing is kept (the other schedulers are unused).

import torch


class WarmupCosineAnnealing:
    def __init__(self, optimizer, max_lr, warmup_rate, T_max, eta_min=0):
        self.warmup_steps = T_max * warmup_rate
        self.current_step=0
        self.cosineannealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max-self.warmup_steps, eta_min)
        self.optimizer = optimizer
        self.max_lr = max_lr

    def step(self):
        if self.current_step < self.warmup_steps:
            # linear wramup
            for g in self.optimizer.param_groups:
                g['lr'] = self.max_lr/self.warmup_steps*self.current_step
        else:
            self.cosineannealing.step()

        self.current_step+=1

    def state_dict(self):
        state_dict = {
            'cosineannealing': self.cosineannealing.state_dict(),
            'current_step': self.current_step,
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.cosineannealing.load_state_dict(state_dict['cosineannealing'])
        self.current_step = state_dict['current_step']
