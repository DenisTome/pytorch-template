# -*- coding: utf-8 -*-
"""
Created on Jan 23 17:31 2019

@author: Denis Tome'

This code is for indexing a dataset generating a single h5 file
containing a list with all the files and their position.

Usage:
    python index_dataset.py -d directory_path [-f format]
"""

import os
import utils
import logging
import argparse

_logger = logging.getLogger('main')

parser = argparse.ArgumentParser(description='Dataset indexer')
parser.add_argument(
    '-i',
    '--path',
    required=True,
    type=str,
    help='path to the directory containing the dataset')
parser.add_argument(
    '-f',
    '--format',
    default='all',
    type=str,
    help='format of the files to be indexed (default: all)')


def index_files(data_path, format):
    """
    Index files according to format
    :param data_path: path to the directory
    :param format
    :return: list of paths and file names
    """
    if len(format) > 5:
        _logger.error('File format cannot be more than 5 characters...')
        exit()

    _format = '*'
    if format != 'all':
        if format.find('.') > -1:
            _format = format.replace('.', '')
        else:
            _format = format

    file_paths, _ = utils.get_files(data_path, _format)

    return file_paths


def main(args):
    # checking that is in the right format
    _data_dir = args.path
    if _data_dir[-1] == '/':
        _data_dir = _data_dir[:-1]

    # making sure that the directory exists
    if not os.path.isdir(args.path):
        _logger.error('Directory {} does not exist...'.format(_data_dir))
        exit()

    # generating list
    _logger.info('Indexing dataset...')
    paths = index_files(_data_dir, args.format)
    _list_utf8 = [utils.encode_str_utf8(p) for p in paths]

    # creating file
    _logger.info('Saving index file...')
    index_path = '{}/index.h5'.format(_data_dir)
    utils.write_h5(index_path, _list_utf8)

    _logger.info('File index.h5 created in {}'.format(_data_dir))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())

