import os

import json
import cv2
print(cv2.__version__)
import argparse
import glob
from transformers import *

import numpy as np



if __name__== "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("in_video", type=str,
                        help = "The video to  annotate")        
    parser.add_argument("out_video", type=str,
                        help = "Path where the ROI video is stored")
    parser.add_argument("--transform", type=str,
                        help="Tells which transform to apply: Default is Hough")


    hTransformer = hough_transform_matlab()    
    args = parser.parse_args()


    transformers = {
        "hough" : hTransformer.step,
        "identity" : identity
    }


    if args.transform is not None:
        assert (args.transform in transformers.keys())
        transformer = transformers[args.transform]
    else:
        transformer = transformers['hough']
    
    transform_frame_by_frame(args.in_video, args.out_video, transformer)



    
    

    
