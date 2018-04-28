import json
import numpy as np
import os
from glob import glob
import sys
import math

from helpers import *

class DataPrepper:
    def __init__(self):
        self.keys  = ["Nose",
                      "Neck" ,
                      "RShoulder",
                      "RElbow",
                      "RWrist",
                      "LShoulder",
                      "LElbow",
                      "LWrist",
                      "RHip",
                      "RKnee",
                      "RAnkle",
                      "LHip",
                      "LKnee",
                      "LAnkle",
                      "REye",
                      "LEye",
                      "REar",
                      "LEar",
                      "Background"]


        self.angles_list =[["LShoulder","LWrist","LElbow"],
                           ["RShoulder","RWrist","RElbow"],
                           ['Nose' ,'LElbow','Neck'],
                           ['Nose' ,'RElbow','Neck']]
    
    def mk_keypoint_dict(self,d):

        keypoint_list = pose_points_2d(d)
        # print( "Keypoint list", keypoint_list)
        return_dict_list = [] 
        # print( "keypoint list", len(keypoint_list))
        for i in keypoint_list:
            return_dict_list.append(dict(zip(self.keys, i)))
        # print( "Rdict", return_dict_list)

        
        return filter (lambda x: len(x.items()) !=0, return_dict_list)

    
        
    def return_features(self, d):
        feature_vector = []
        if d != []:            
            for i in self.angles_list:
                # print ("Dict", d)
                feature_vector.append(angle_calc(d[i[0]],
                                                d[i[1]],
                                                d[i[2]]))

            #print ("Return features", feature_vector)                
            if not np.all(np.isnan(feature_vector)):
                # print ("Correct Ret")
                return np.array(feature_vector)
            else: 
                 return np.repeat(np.nan, len(self.angles_list))
            
            
    def single_step(self, json_file_path):
        with open(json_file_path) as f:
            keypoint_json = json.load(f)
            keypoint_dict_list = self.mk_keypoint_dict(keypoint_json)
            return_features_list = []
            for i in keypoint_dict_list:
                fv = self.return_features(i)                
                return_features_list.append(fv)
            # print( "Return feature list", return_features_list)
            return np.vstack(return_features_list)
            

        
    def multi_step(self, json_file_dir):
        print (os.path.abspath(json_file_dir))
        assert(os.path.isdir(os.path.abspath(json_file_dir)))
        feature_vector_list = []
        for i in sorted(glob(os.path.join(os.path.abspath(json_file_dir),
                      "*.json"))):
            # print ("Processing file : ", i
            feature_vector_list.append(self.single_step(i))
                                      
        r_val =  np.vstack(feature_vector_list)
        return r_val[~np.isnan(r_val).all(axis=1)]




    

if __name__=="__main__":    
    dataPrepper = DataPrepper()    
    f =  '/home/akash/learn/504/final_project/EECS_504_Project/data_json/IMG_20180406_193530_000000000000_keypoints.json'
    #f = '/home/akash/learn/504/final_project/EECS_504_Project/openpose/sample1_pose/sample1_000000000051_keypoints.json'
    
    
    
    directory = "/home/akash/learn/504/final_project/EECS_504_Project/data_json/"
    x = dataPrepper.multi_step(directory)
    for i in x:
        print ("Feature Vector :", i)


        
