print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.preprocessing import Imputer

from pepper import DataPrepper
import argparse

if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("in_video", type=str,
                        help = "The original video that needs to be annotated")
    parser.add_argument("in_json", type=str,
                        help = "The JSON directory on the region of interest")
    parser.add_argument("out_video", type=str,
                        help = "Path where the ROI video is stored")

    args = parser.parse_args()

    
    ### TRAIN PHASE 
    # List of data files.
    train_dir_dict = {
        'data_lht_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_lht_r_json',
        'data_lht_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_lht_l_json',
        'data_looking_away_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_away_r_json',
        'data_looking_away_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_away_l_json',
        'data_looking_forward_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_forward_l_json',
        'data_looking_forward_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_forward_r_json',
        'data_looking_down_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_down_r_json',
        'data_looking_down_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_looking_down_l_json',
        'data_rht_r' :'/home/akash/learn/504/final_project/EECS_504_Project/data_rht_r_json',
        'data_rht_l' :'/home/akash/learn/504/final_project/EECS_504_Project/data_rht_l_json'}
    
    svm_dict = {}
    
    train_directory = '/home/akash/learn/504/final_project/EECS_504_Project/data_train_json'
    test_directory = '/home/akash/learn/504/final_project/EECS_504_Project/data_bad_json'


    # Pepper will take care of getting the data in a nice insidious format
    pepper = DataPrepper()
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    
    for k,v in train_dir_dict.items():
        train_data = []
        train_data = pepper.multi_step(v)
        clf = svm.OneClassSVM(nu = 0.1, kernel='poly')
        train_data_imp = imp.fit_transform(train_data)    
        clf.fit(train_data_imp)
        svm_dict[k] = clf
        y_train_pred = clf.predict(train_data_imp)
        # print ("Train array len", len(y_train_pred))    
        # print ("Train diff", (sum(y_train_pred)))


    

    
# test_directory = '/home/akash/learn/504/final_project/EECS_504_Project/data_bad_json'

# y_test = pepper.multi_step(test_directory)
# y_test_file = '/home/akash/learn/504/final_project/EECS_504_Project/data_train_json/4519661B00000578-4955690-image-a-23_1507304927988_000000000000_keypoints.json'
# y_test_file = '/home/akash/learn/504/final_project/EECS_504_Project/data_train_json/8_000000000000_keypoints.json'
# y_test_file = '/home/akash/learn/504/final_project/EECS_504_Project/data_train_json/IMG_20180406_193542_000000000000_keypoints.json'
# y_test_file =  '/home/akash/learn/504/final_project/EECS_504_Project/test_img_json/IMG_20180408_010659_000000000000_keypoints.json'
# y_test = pepper.single_step(y_test_file)
# print ("Y_test", y_test)
# y_test_imp = imp.fit_transform(y_test)
# for k,v in svm_dict.items():
#     y_test_pred = v.predict(y_test)
#     print ("Type : ", k, y_test_pred)
    




# train_data_1 = train_pepper.multi_step(train_directory)
# train_data_2 = train_pepper.multi_step('/home/akash/learn/504/final_project/EECS_504_Project/data_video_bad_json/.')

# train_data = np.vstack((train_data_1, train_data_2))

# test_data = train_pepper.multi_step(test_directory)

# #good one here: clf = svm.OneClassSVM(nu=0.19, kernel="poly")
# clf = svm.OneClassSVM(nu=0.1, kernel="poly")
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# train_data_imp = imp.fit_transform(train_data)
# test_data_imp = imp.fit_transform(test_data)

# clf.fit(train_data_imp)


# y_train_pred = clf.predict(train_data_imp)
# y_test_pred = clf.predict(test_data_imp)

# print ("Train array", (y_train_pred))
# print ("Train diff", (sum(y_train_pred)))

# print ("Test array", (y_test_pred))
# print ("Test diff", (sum(y_test_pred)))












# xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# # Generate train data
# X = 0.3 * np.random.randn(100, 2)
# X_train = np.r_[X + 2, X - 2]
# # Generate some regular novel observations
# X = 0.3 * np.random.randn(20, 2)
# X_test = np.r_[X + 2, X - 2]
# # Generate some abnormal novel observations
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# # fit the model
# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# clf.fit(X_train)
# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)
# n_error_train = y_pred_train[y_pred_train == -1].size
# n_error_test = y_pred_test[y_pred_test == -1].size
# n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# # plot the line, the points, and the nearest vectors to the plane
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.title("Novelty Detection")
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
# a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

# s = 40
# b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
#                  edgecolors='k')
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
#                 edgecolors='k')
# plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.legend([a.collections[0], b1, b2, c],
#            ["learned frontier", "training observations",
#             "new regular observations", "new abnormal observations"],
#            loc="upper left",
#            prop=matplotlib.font_manager.FontProperties(size=11))
# plt.xlabel(
#     "error train: %d/200 ; errors novel regular: %d/40 ; "
#     "errors novel abnormal: %d/40"
#     % (n_error_train, n_error_test, n_error_outliers))
# plt.show()
