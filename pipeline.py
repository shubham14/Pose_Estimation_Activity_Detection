import os

import json
import cv2
print(cv2.__version__)
import argparse
import glob
from helpers import *


def get_jsons(json_dir):
    pose_frames_list = []
    json_file_list = sorted(list(glob.glob(os.path.join(json_dir, "*.json"))))

    for i in json_file_list:
        with open(i) as f:
            pose_frames_list.append(json.loads( f.read()))

    return pose_frames_list




def annotate_vid_main(in_vid_name,
                      out_vid_name,
                      json_dir):
    pose_frames_list = get_jsons(json_dir)

    assert (len(pose_frames_list) != 0)

    # Set the input video handle
    in_vid_handle = cv2.VideoCapture(in_vid_name)
    # Get the first frame
    success,image = in_vid_handle.read()

    # Get the dimensions of the image to use it for the out_video
    # parameterization
    height, width, layers = image.shape
    count = 0
    success = True

    ##  Additional Parameters for he vidoe writer
    # forcc = cv2.VideoWriter_fourcc('F','M','P','4')
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Open the output video handle 
    out_video_handle = cv2.VideoWriter(out_video_name,
                                       forcc,
                                       16,
                                       (int(in_vid_handle.get(3)), int(in_vid_handle.get(4))))
    
    while success:
        # Get the curr_pose
        curr_pose = pose_frames_list[count]
        # Get the pose_points_2d
        coor = pose_points_2d(curr_pose)

        for point in coor:
            print (point)
            x =  tuple(map(int,point[0:2]))
            image = cv2.circle(image,x,1,(0,0,255))
        out_video_handle.write(image)
    
    success,image = in_vid_handle.read()
    print ('Read a new frame: ', success)
    count += 1

        
    cv2.destroyAllWindows()
    out_video_handle.release()


if __name__== "__name__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("out_video", type=str,
                        help = "Path where the annotated video should be stored")
    parser.add_argument("in_video", type=str,
                        help = "The video to  annotate")    
    parser.add_argument("json_dir", type=str,
                        help="json directory of frame information for the in video")
    
