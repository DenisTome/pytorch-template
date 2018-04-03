from utils.util import ensure_dir
from tensorboardX import SummaryWriter


class Logger(SummaryWriter):
    """ Logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self, dir_path):
        super().__init__()
        self.dir_path = dir_path
        ensure_dir(self.dir_path)

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

