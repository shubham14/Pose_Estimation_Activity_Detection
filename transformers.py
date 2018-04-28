import subprocess
import cv2
print(cv2.__version__)
import os
import sys
import matlab.engine


import uuid

def identity(frame):
    return frame


class hough_transform_matlab(object):
    def __init__(self):
        self.execute = matlab.engine.start_matlab()

    def step(self,image_array):
        # Get unique names
        to_hough_frame =  str(uuid.uuid4()) + '.png'
        from_hough_frame = str(uuid.uuid4()) +   '.png'

        # Write image
        cv2.imwrite(to_hough_frame, image_array)

        self.execute.proc_image(to_hough_frame, from_hough_frame,nargout=0)

        img = cv2.imread(from_hough_frame)
        os.remove(to_hough_frame)
        os.remove(from_hough_frame)
        return img
        

    
                       

def transform_frame_by_frame(in_vid_name,
                             out_vid_name,
                             transform_frame):

    # Set the input video handle
    in_vid_handle = cv2.VideoCapture(in_vid_name)
    # Get the first frame
    success, image = in_vid_handle.read()
    print ("Image", type(image), image.shape)
    fps = in_vid_handle.get(cv2.CAP_PROP_FPS)
    print (fps)
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
    out_video_handle = cv2.VideoWriter(out_vid_name,
                                       fourcc,
                                       fps,
                                       (int(in_vid_handle.get(3)), int(in_vid_handle.get(4))))
    
    while success:
        transformed_image = transform_frame(image)

        out_video_handle.write(transformed_image)
    
        success,image = in_vid_handle.read()
        count += 1

        
    cv2.destroyAllWindows()
    out_video_handle.release()
    in_vid_handle.release()
