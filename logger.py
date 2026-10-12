try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
 
 
class Logger:
    """Same interface as the reference repo's logger. If tensorboard isn't
    installed, logging is silently skipped instead of crashing training."""
 
    def __init__(self, log_dir):
        if SummaryWriter is None:
            print("tensorboard not installed - skipping TensorBoard logging")
            self.writer = None
        else:
            self.writer = SummaryWriter(log_dir)
 
    def scalar_summary(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
 
    def close(self):
        if self.writer is not None:
            self.writer.close()
