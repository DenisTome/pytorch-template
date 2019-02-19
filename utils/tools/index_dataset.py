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
import logging
import argparse
import utils

_LOGGER = logging.getLogger('main')

PARSER = argparse.ArgumentParser(description='Dataset indexer')
PARSER.add_argument(
    '-i',
    '--path',
    required=True,
    type=str,
    help='path to the directory containing the dataset')
PARSER.add_argument(
    '-f',
    '--format',
    default='all',
    type=str,
    help='format of the files to be indexed (default: all)')


def index_files(data_path, file_format):
    """Index files for faster dataset loading time

    Arguments:
        data_path {str} -- path to directory containing files
        file_format {str} -- format of the files that are indexed

    Returns:
        list -- file paths
    """

    if len(file_format) > 5:
        _LOGGER.error('File format cannot be more than 5 characters...')
        exit()

    search_format = '*'
    if file_format != 'all':
        if file_format.find('.') > -1:
            search_format = file_format.replace('.', '')
        else:
            search_format = file_format

    file_paths, _ = utils.get_files(data_path, search_format)

    return file_paths


def main(args):
    """Main"""

    # checking that is in the right format
    data_dir = args.path
    if data_dir[-1] == '/':
        data_dir = data_dir[:-1]

    # making sure that the directory exists
    if not os.path.isdir(args.path):
        _LOGGER.error('Directory %s does not exist...', data_dir)
        exit()

    # generating list
    _LOGGER.info('Indexing dataset...')
    paths = index_files(data_dir, args.format)
    list_utf8 = [utils.encode_str_utf8(p) for p in paths]

    # creating file
    _LOGGER.info('Saving index file...')
    index_path = '{}/index.h5'.format(data_dir)
    utils.write_h5(index_path, list_utf8)

    _LOGGER.info('File index.h5 created in %s', data_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(PARSER.parse_args())
