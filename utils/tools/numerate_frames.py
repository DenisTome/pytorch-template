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
import logging
import argparse
import cv2
import utils


_FILE_FORMATS = ['jpg', 'png']
_COLORS = {'b': (0, 0, 0),
           'w': (255, 255, 255)}
_ALIGNMENT = {'t': 0.1, 'b': 0.9,
              'c': 0.5, 'l': 0.1, 'r': 0.9}

_LOGGER = logging.getLogger('main')

PARSER = argparse.ArgumentParser(description='Numerate frames')
PARSER.add_argument(
    '-i',
    '--input',
    required=True,
    type=str,
    help='input directory containing all images')
PARSER.add_argument(
    '-o',
    '--output',
    required=True,
    type=str,
    help='output directory where to save the numerated frames')
PARSER.add_argument(
    '-c',
    '--color',
    default='b',
    type=str,
    help='color {w, b}')
PARSER.add_argument(
    '--pos',
    default='tl',
    type=str,
    help='position of the name {tl, tc, tr, bl, bc, br}')


def get_position(img, pos):
    """Get location for the image name
    in the frame image.

    Arguments:
        img {numpy array} -- matrix containing the image
        pos {str} -- position of the image name

    Returns:
        (u,v) -- pixel coordinates
    """

    h, w = img.shape[:2]
    v = int(_ALIGNMENT[pos[0]] * h)
    u = int(_ALIGNMENT[pos[1]] * w)
    return (u, v)


def annotate_images(path, output_dir, color, align):
    """Annotate all images with image number

    Arguments:
        path {str} -- path to the dir containing the images
        output_dir {str} -- path where to save the images
        color {str} -- color for the text
        align {str} -- alignment for the text
    """

    _LOGGER.info('Indexing images...')
    utils.ensure_dir(output_dir)
    paths, names = utils.get_files(path, _FILE_FORMATS)

    # set properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = _COLORS[color]
    line_type = 2

    _LOGGER.info('Saving new images...')
    for img_path, img_name in zip(paths, names):
        output_path = img_path.replace(path, output_dir)
        img = cv2.imread(img_path)
        text_pos = get_position(img, align)

        cv2.putText(img,
                    img_name,
                    text_pos,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        cv2.imwrite(output_path, img)


def main(args):
    """Main"""

    if not (args.pos in ['tl', 'tc', 'tr', 'bl', 'bc', 'br']):
        _LOGGER.error('Position for the file name not recognized...')
        exit()

    if (args.color in ['b', 'w']):
        _LOGGER.error('Color for the text not recognized...')
        exit()

    annotate_images(args.input,
                    args.output,
                    args.color,
                    args.pos)
    _LOGGER.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(PARSER.parse_args())
