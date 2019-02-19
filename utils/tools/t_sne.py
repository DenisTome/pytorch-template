# -*- coding: utf-8 -*-
"""
Created on Jan 24 11:54 2019

@author: Denis Tome'

This allows to project a large dimensional space into
2D to be able to plot the space, using as a projection
tSNE.

"""
import re
import os
import logging
import argparse
import numpy as np
from sklearn.manifold import TSNE
from base.template import FrameworkClass
import utils


_LOGGER = logging.getLogger('main')

PARSER = argparse.ArgumentParser(description='tSNE data projector')
PARSER.add_argument(
    '-i',
    '--file-path',
    required=True,
    type=str,
    help='path to the file containing the data (supported npy or h5)')


class TSNEProjector(FrameworkClass):
    """Class for TSNE projection"""

    def __init__(self, dim=2):
        super().__init__()

        self.num_dimension = dim
        self.projector = TSNE(n_components=self.num_dimension)

    def fit_project(self, data):
        """Fit the model on the N-dimensional data
        and project them on a lower dimensional space

        Arguments:
            data {numpy array} -- format NUM_SAMPLES x DIMENSIONALITY

        Returns:
            numpy array -- format NUM_SAMPLES x 2
        """

        return self.projector.fit_transform(data)

    def fit(self, data):
        """Fit model on the N-dimensional data

        Deprecated: use fit_project

        Arguments:
            data {numpy array} -- format NUM_SAMPLES x 2
        """

        raise NotImplementedError


def _read_data(path):
    # checking that the file is in the right format

    file_format = re.findall('\.(\w+)', path)[-1]
    if not file_format in ['npy', 'h5']:
        _LOGGER.error("File type not supported. Only npy and h5 files...")
        exit()

    # reading data content
    _LOGGER.info('Reading data...')
    if file_format == 'h5':
        data = utils.read_h5(path)['val']
    else:
        data = np.load(path)

    if data.ndim != 2:
        _LOGGER.error('Data matrix is not (N x M)')
        exit()

    _LOGGER.info('Projecting data. This might take a while...')
    proj = TSNEProjector()
    return proj.fit_project(data)


def main(args):
    """Main"""

    # checking that the file exists
    file_path = args.file_path
    if not os.path.exists(file_path):
        _LOGGER.error('File {} does not exist'.format(file_path))
        exit()

    # processing data
    data = _read_data(file_path)

    # creating output file
    _LOGGER.info('Saving output data in file...')
    out_dir = file_path.replace(re.findall(r'\w+\.\w+', file_path)[-1], '')
    out_file = '{}output.h5'.format(utils.ensure_dir(out_dir))
    utils.write_h5(out_file, data)

    _LOGGER.info('File output.h5 created in %s', out_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(PARSER.parse_args())
