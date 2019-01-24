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
import utils
import logging
import argparse
import numpy as np
from base.template import FrameworkClass
from sklearn.manifold import TSNE


_logger = logging.getLogger('main')

parser = argparse.ArgumentParser(description='tSNE data projector')
parser.add_argument(
    '-i',
    '--file-path',
    required=True,
    type=str,
    help='path to the file containing the data (supported npy or h5)')


class TSNE_projector(FrameworkClass):

    def __init__(self, dim=2):
        super().__init__()

        self.num_dimension = dim
        self.projector = TSNE(n_components=self.num_dimension)

    def fit_project(self, data):
        """
        Fit the model on the N-dimensional data and project
        them onto a lower dimensional space
        :param data: format (NUM_SAMPLES x DIMENSION)
        :return: (NUM_SAMPLES x LOWER_DIMENSION)
        """
        return self.projector.fit_transform(data)

    def fit(self, data):
        """
        Fit model on the N-dimensional data
        :param data: format (NUM_SAMPLES x DIMENSION)
        :return: (NUM_SAMPLES x LOWER_DIMENSION)
        """
        self.projector.fit(data)
        self._logger.warning("Functionality not usable...")


def _read_data(path):
    # checking that the file is in the right format

    _format = re.findall('\.(\w+)', path)[-1]
    if not _format in ['npy', 'h5']:
       _logger.error("File type not supported. Only npy and h5 files...")
       exit()

    # reading data content
    _logger.info('Reading data...')
    if _format == 'h5':
        data = utils.read_h5(path)['val']
    else:
        data = np.load(path)

    if data.ndim != 2:
        _logger.error('Data matrix is not (N x M)')
        exit()

    _logger.info('Projecting data. This might take a while...')
    proj = TSNE_projector()
    return proj.fit_project(data)


def main(args):
    # checking that the file exists
    _path = args.file_path
    if not os.path.exists(_path):
        _logger.error('File {} does not exist'.format(_path))
        exit()

    # processing data
    data = _read_data(_path)

    # creating output file
    _logger.info('Saving output data in file...')
    _out_dir = _path.replace(re.findall('\w+\.\w+', _path)[-1], '')
    _out_file = '{}output.h5'.format(utils.ensure_dir(_out_dir))
    utils.write_h5(_out_file, data)

    _logger.info('File output.h5 created in {}'.format(_out_dir))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())