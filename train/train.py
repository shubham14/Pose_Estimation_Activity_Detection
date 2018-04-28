import glob
import argparse

import sys
import glob
import os
import string

from functools import reduce

template_str = string.Template("""cd openpose; 
build/examples/openpose/openpose.bin --render_pose 1  --face_render 0  --hand_render 0  --video $in_image  --display 0  -write_json $json_dir  --write_images $out_image_dir -number_people_max 1; 
cd -""")




if __name__== "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("in_image_dir", type=str,
                        help = "The input images folder ")
    

    parser.add_argument("json_dir", type=str,
                        help="json directory output directory")

    parser.add_argument("out_image_dir", type=str,
                        help = "The output images folder")


    args = parser.parse_args()
    tdict = {}
    tdict["out_image_dir"] = args.out_image_dir
    tdict["json_dir"] = args.json_dir

    patterns = ['*.png' ,'*.jpg', '*.jpeg', '*.mp4']
    search_path =  [os.path.join(os.path.abspath(args.in_image_dir), i) for i in patterns]
    glob_list =  reduce(lambda x,y : x+y, [glob.glob(e) for e in search_path])


    for i in glob_list:
        tdict['in_image'] = i
        print (template_str.substitute(tdict))
        

    

    
    


    
    

    
