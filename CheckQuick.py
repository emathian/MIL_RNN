import json
import cv2
import argparse
parser = argparse.ArgumentParser(description='Check img validuty in set.')

parser.add_argument('--x', type=int,    help="Input directory where the images are stored")
parser.add_argument('--y', type=int,    help="Input directory where the images are stored")
args = parser.parse_args()

y = args.y
x = args.x 
with open('TrainNormalTumoral.json') as json_file:
    lib = json.load(json_file)
slides = lib['Slides'][round(len(lib['Slides']) * x / 100 ):round(len(lib['Slides']) * y / 100 )]
print(len(slides))
print(round(len(lib['Slides']) * x / 100 ))
print(round(len(lib['Slides']) * y / 100 ))
print(slides, '\n')
tot_number_of_files = 0
TilesM = lib['Tiles'][round(len(lib['Slides']) * x / 100 ):round(len(lib['Slides']) * y / 100 )]
for i in TilesM:
    tot_number_of_files += len(i)
tot_number_of_files
c = 0
for IDX in range(len(TilesM)):
    Tiles = TilesM[IDX]
    print(slides[IDX])     
    for t in Tiles:
        c += 1
        if c * 100 / tot_number_of_files % 5 == 0:
            print('Process {}'.format( c * 100 / tot_number_of_files ))
        try:
            if cv2.imread(t).shape !=  None:
                n = 0
            else:
                print('Error with {}'.format(t))
        except:
            print('Error with {}'.format(t))
