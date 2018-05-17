from os.path import dirname, abspath, join
# import socket

__all__ = ['ROOT_DIR', 'DATA_DIR', 'CHKP_DIR']

# defining relative paths
CURR_DIR = dirname(abspath(__file__))
ROOT_DIR = join(CURR_DIR, '../')
DATA_DIR = join(ROOT_DIR, 'data/')
CHKP_DIR = join(DATA_DIR, 'checkpoints/')

# if socket.gethostname() == 'denistome-mpb':
# add machine specific paths
