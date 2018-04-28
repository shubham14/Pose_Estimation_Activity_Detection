import numpy as np

def get_angle(a,b,c):

    np_a = np.array(list(a[:2])) 
    np_b = np.array(list(b[:2])) 
    np_c = np.array(list(c[:2]))
    ba = np_a - np_b
    bc = np_c - np_b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    if c[2] == 0 or b[2] == 0 or c[2] == 0:
        return np.nan
    else:
        return np.degrees(angle)


# function to angle, where the input arguments are a tuple
def angle_calc(t1, t2, t_basis):
    x1, y1 = t1[0], t1[1]
    n1 = np.array([x1, y1])
    x2, y2 = t2[0], t2[1]
    n2 = np.array([x2, y2]) 
    x_basis, y_basis = t_basis[0], t_basis[1]
    n_basis = np.array([x_basis, y_basis])
    na = n1 - n_basis
    nb = n2 - n_basis
    cosine_angle = np.dot(nb, na) / (np.linalg.norm(na) * np.linalg.norm(nb))
    angle = np.arccos(cosine_angle)
    if t1[2] != 0 and t2[2] != 0:
        return np.degrees(angle) 
    else:
        return np.nan



    
def zigzag(seq):
   return list(zip(seq[::3], seq[1::3],seq[2::3]))

def pose_points_2d(openpose_dict):
   coordinates = []
   for people in openpose_dict['people']:
       # print ((zigzag(people['pose_keypoints_2d'])))
       coordinates = coordinates + [(zigzag(people['pose_keypoints_2d']))]
   return coordinates

