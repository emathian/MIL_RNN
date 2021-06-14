import json
import cv2
import argparse
parser = argparse.ArgumentParser(description='Check img validuty in set.')

parser.add_argument('--folder', type=str,    help="Input directory where the images are stored")
parser.add_argument('--split', type=str,    help="Input directory where the images are stored")
args = parser.parse_args()
folder = args.folder
split = args.split

import json
with open('/home/mathiane/MIL-nature-medicine-2019/{}.json'.format(split)) as json_file:
            lib = json.load(json_file)
slides = lib['Slides']

IDX = -1

for i, s in enumerate(slides):
    if s == folder:
        IDX = i
if IDX != -1:
    Tiles = lib['Tiles'][IDX]     
    with open('error_tiles_{}_{}.txt'.format(split, folder), 'a') as f:
        for t in Tiles:
            try:
                if cv2.imread(t).shape !=  None:
                    c = 0
                else:
                    f.write('{}\n'.format(t))
            except:
                f.write('{}\n'.format(t))
