from torch import optim


def get_scheduler(optimizer, num_epochs, lr_args):
        lr_policy = lr_args['lr_policy']
        if lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + lr_args['epoch_count'] - num_epochs) / float(lr_args['n_epochs_decay'] + 1) # start epoch는 1로?
                return lr_l
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_args['lr_decay_iters'], gamma=0.1)
        elif lr_policy == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif lr_policy == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', lr_args['lr_policy'])

        return scheduler