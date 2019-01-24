# -*- coding: utf-8 -*-
"""
Created on Jan 24 15:54 2019

@author: Denis Tome'

This code add the frame number to each frame in the input directory
which is very useful when we want to create a video out of them.
In this way, if we want to stop the video at a specific point to analyze
something we know exactly what is the frame number corresponding to the
visualized frame.

"""
import cv2
import utils
import logging
import argparse

_FILE_FORMATS = ['jpg', 'png']
_logger = logging.getLogger('main')

parser = argparse.ArgumentParser(description='Numerate frames')
parser.add_argument(
    '-i',
    '--input',
    required=True,
    type=str,
    help='input directory containing all images')
parser.add_argument(
    '-o',
    '--output',
    required=True,
    type=str,
    help='output directory where to save the numerated frames')


def annotate_images(path, output_dir):
    """
    Annotate all images with image number
    """
    _logger.info('Indexing images...')
    paths, names = utils.get_files(path, _FILE_FORMATS)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 750)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 1

    _logger.info('Saving new images...')
    for img_path, img_name in zip(paths, names):
        output_path = img_path.replace(path, output_dir)
        img = cv2.imread(img_path, 0)

        cv2.putText(img,
                    img_name,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imwrite(output_path, img)


def main(args):
    annotate_images(args.input,
                    args.output)
    _logger.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args())
