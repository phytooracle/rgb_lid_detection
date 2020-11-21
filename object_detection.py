#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2020-11-06
Purpose: RGB lid detection
"""

import argparse
from multiprocessing import process
import os
import sys
from detecto import core, utils, visualize
import glob
import cv2
from detecto.core import Model
import numpy as np
import tifffile as tifi
from osgeo import gdal
import pyproj
import utm
import json
import pandas as pd
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='RGB lid detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dir',
                        metavar='dir',
                        help='Directory containing TIFF images')

    parser.add_argument('-m',
                        '--model',
                        help='A .pth model file',
                        metavar='model',
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument('-o',
                        '--outdir',
                        help='Output directory',
                        metavar='outdir',
                        type=str,
                        default='detect_out')

    args = parser.parse_args()

    if '/' not in args.dir[-1]:
        args.dir = args.dir + '/'

    return args

# --------------------------------------------------
def get_min_max(box):
    min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    return min_x, min_y, max_x, max_y


# --------------------------------------------------
def pixel2geocoord(one_img, x_pix, y_pix):
    ds = gdal.Open(one_img)
    c, a, b, f, d, e = ds.GetGeoTransform()
    lon = a * int(x_pix) + b * int(y_pix) + a * 0.5 + b * 0.5 + c
    lat = d * int(x_pix) + e * int(y_pix) + d * 0.5 + e * 0.5 + f

    return (lat, lon)


# --------------------------------------------------
def open_image(img_path):

    args = get_args()

    a_img = tifi.imread(img_path)
    a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
    a_img = np.array(a_img)

    return a_img


# --------------------------------------------------
def process_image(img):
    args = get_args()
    cont_cnt = 0
    lett_dict = {}

    model = core.Model.load(args.model, ['lid'])

    a_img = open_image(img)
    df = pd.DataFrame()
    try:
        predictions = model.predict(a_img)
        labels, boxes, scores = predictions
        print(f'Image: {img}\nPredictions: {boxes}\nScores: {scores}')
        copy = a_img.copy()

        for i, box in enumerate(boxes):
            if scores[i] >= 0.5:
                cont_cnt += 1

                min_x, min_y, max_x, max_y = get_min_max(box)
                center_x, center_y = ((max_x+min_x)/2, (max_y+min_y)/2)
                lett_dict[cont_cnt] = {
                    'image': img,
                    'center_x': int(center_x),
                    'center_y': int(center_y),
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y
                }

        df = pd.DataFrame.from_dict(lett_dict, orient='index', columns=['image',
                                                                    'center_x',
                                                                    'center_y',
                                                                    'min_x',
                                                                    'max_x',
                                                                    'min_y',
                                                                    'max_y']).set_index('image')
    except:
        pass

    return df


# --------------------------------------------------
def main():
    """Detect lid/s in images"""

    args = get_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    img_list = glob.glob(f'{args.dir}*.tif')
    major_df = pd.DataFrame()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        df = p.map(process_image, img_list)
        major_df = major_df.append(df)

    out_path = os.path.join(args.outdir, f'lid_detection.csv')
    major_df.to_csv(out_path)

    print(f'Done, see outputs in ./{args.outdir}.')


# --------------------------------------------------
if __name__ == '__main__':
    main()
