def zigzag(seq):
    return list(filter( lambda x: x[2] != 0 ,zip(seq[::3], seq[1::3],seq[2::3])))
    
        
    
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
