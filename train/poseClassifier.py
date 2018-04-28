import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.preprocessing import Imputer
from pepper import DataPrepper
import cv2
from glob import glob
import json
from collections import OrderedDict
import operator
from helpers import *
def zigzag(seq):
    return list(zip(seq[::3], seq[1::3],seq[2::3]))



def pose_points_2d(openpose_dict):    
    coordinates = []
    for people in openpose_dict['people']:
        coordinates = coordinates + (zigzag(people['pose_keypoints_2d']))
    return coordinates


def pose_points_2d_mapped(openpose_dict):
    keys  = ["Nose",
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
    keypoints_list = pose_points_2d(openpose_dict)
    return dict(zip(keys, keypoints_list))


class PoseClassifier:


    def __init__(self):
        self.threshold = 0.5
        ### TRAIN PHASE 
        # List of data files.
        self.train_dir_dict_1 = {
            'data_lht_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_lht_r_json',
            'data_lht_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_lht_l_json',
            # 'data_lht' :'/home/akash/learn/504/final_project/EECS_504_Project/data_lht_json',
            'data_looking_away_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_away_r_json',
            'data_looking_away_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_away_l_json',
            # 'data_looking_forward_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_forward_l_json',
            # 'data_looking_forward_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_forward_r_json',
            'data_looking_down_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_down_r_json',
            'data_looking_down_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_down_l_json',
            'data_rht_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_rht_r_json',
            'data_rht_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_rht_l_json'}
            #            'data_rht' :'/home/akash/learn/504/final_project/EECS_504_Project/data_rht_json'}

        self.train_dir_dict =  OrderedDict(sorted(self.train_dir_dict_1.items(), key=lambda t: t[0]))

        self.svm_dict = {}





        # Pepper will take care of getting the data in a nice insidious format
        self.pepper = DataPrepper()
        self.imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    
        for k,v in self.train_dir_dict.items():
            train_data = []
            # print ("V is ", v)                
            train_data = self.pepper.multi_step(v)
            self.clf = svm.OneClassSVM(nu = 0.1, kernel='poly')
            train_data_imp = self.imp.fit_transform(train_data)    
            self.clf.fit(train_data_imp)
            self.svm_dict[k] = self.clf
            y_train_pred = self.clf.predict(train_data_imp)
            print ("Train array len", len(y_train_pred))    
            print ("Train diff", (sum(y_train_pred)))

    def multi_step(self, in_json_dir):
        test_data = self.pepper.multi_step(in_json_dir)
        # print ("Test Data", test_data.shape)
        #test_data_imp = test_data
        #test_data_imp[inds] = 300.0
        
        test_data_imp = self.imp.fit_transform(test_data)
        results = [] 
        for k,v in self.svm_dict.items():                        
            results.append( v.predict(test_data_imp) )

            
        return np.vstack(results).T, list(self.train_dir_dict.keys())


    

    def get_coordinates(self,json_path, thresh):
        #  top coordinates of the man
        
        json_dict = pose_points_2d_mapped(json.loads(open(json_path).read()))
        # print (json_dict.keys())
        (x , y, conf) = json_dict['Neck']
        if conf > np.abs(thresh):
            coordinates = (x, y)
        else:
            (x , y, conf) = json_dict['Nose']
            coordinates = (x, y - 5)
        return coordinates

    def check_if_active(self, json_path, val_dict, label):
        json_dict = pose_points_2d_mapped(json.loads(open(json_path).read()))
        assert(os.path.isfile(json_path))
        if label == 'data_looking_away_l' or label == 'data_looking_away_r':            
            if json_dict['REye'][2] > 0.0 and json_dict['LEye'][2] > 0.0:
                # print (list(json_dict.items()))
                return -3

        if label == 'data_rht_l' or label == 'data_rht_r':
            #print ("Angle", angle_calc(json_dict["RShoulder"],json_dict["RWrist"],json_dict["RElbow"]))                
            if (angle_calc(json_dict["RShoulder"],json_dict["RWrist"],json_dict["RElbow"]) > 60 or
                angle_calc(json_dict["RShoulder"],json_dict["RWrist"],json_dict["RElbow"]) < 5 or
                json_dict["RShoulder"][2] == 0 or
                json_dict["RWrist"][2] == 0 or 
                json_dict["RElbow"][2] == 0
            ):
                return -3
            else:
                return 1



        if label == 'data_lht_l' or label == 'data_lht_r':
            if (angle_calc(json_dict["LShoulder"],json_dict["LWrist"],json_dict["LElbow"]) > 60 or
                angle_calc(json_dict["LShoulder"],json_dict["LWrist"],json_dict["LElbow"]) < 5 or
                json_dict["LShoulder"][2] == 0 or
                json_dict["LWrist"][2] == 0 or 
                json_dict["LElbow"][2] == 0):
                return -3

            else:
                return 1
            # if val_dict['data_rht_l'] == -1 or  val_dict['data_rht_r'] == -1:
        #         return -3
            
        # if label == 'data_lht_l' or label == 'data_lht_r':
        #     if  val_dict['data_lht_l'] == -1 or val_dict['data_lht_r'] == -1:
        #         return -3

        if label == 'data_looking_down_l':
            if (angle_calc(json_dict["LShoulder"],json_dict["LWrist"],json_dict["LElbow"]) > 120  or
                json_dict['LWrist'][2] == 0 or
                json_dict['LShoulder'][2] == 0 or
                json_dict['LElbow'][2] == 0):
                return -3
            else:
                return 0

        if label == 'data_looking_down_r':
            if (angle_calc(json_dict["RShoulder"],json_dict["RWrist"],json_dict["RElbow"]) > 120 or
                json_dict['RWrist'][2] == 0 or
                json_dict['RShoulder'][2] == 0 or
                json_dict['RElbow'][2] == 0):
                return -3
            else :
                return 0
            
            return -3
        return 0
                
    def annotate(self, in_img_dir, in_json_dir,out_img_dir):
        txt_size = 0.5
        val, labels = self.multi_step(in_json_dir)
        print ("val_dict length", val.shape)
        for cnt, (image_path, json_path) in enumerate(zip (sorted(glob(os.path.join(os.path.abspath(in_img_dir),
                                                                             "*.png"))),
                                                    sorted(glob(os.path.join(os.path.abspath(in_json_dir),
                                                                              "*.json"))))):
            if cnt >= val.shape[0]:
                break
            image = cv2.imread(image_path)
            x,y,l = image.shape
            coor = self.get_coordinates(json_path, self.threshold)
            # print ("Cnt is", cnt)
            

            val_dict = dict(zip(labels,val[cnt]))

            for i in range(len(labels)):
                    
                ind = (txt_size, txt_size + 20 * i)
                loc = tuple(map(operator.add, coor,ind))
                # print ("labels", labels[i])
                image = cv2.putText(image,
                                    text = labels[i],
                                    org = tuple(map(int,loc)) ,
                                    fontFace=3,
                                    fontScale = txt_size,
                                    color = (0,
                                             255,
                                             0)  if val_dict[labels[i]] + self.check_if_active(json_path, val_dict, labels[i] ) + 1 > 0 else (0,0,255),
                                    thickness = 1)
            # print ("Writing,", os.path.join(out_img_dir,image_path))
            cv2.imwrite( os.path.join(out_img_dir,os.path.basename(image_path)),
                         image );

