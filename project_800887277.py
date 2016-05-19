import csv
import glob
import os
import cv2
import numpy as np
from sklearn import decomposition, linear_model, cross_validation
from sklearn.metrics import log_loss
from skimage.feature import hog
from sklearn.cross_validation import train_test_split

# location of folders where the images and csv exist
train_image_folder = 'C:/Users/ssirigin/Desktop/MachineLearning project/project/train/'
test_image_folder = 'C:/Users/ssirigin/Desktop/MachineLearning project/project/test/'
train_csv_folder = 'C:/Users/ssirigin/Desktop/MachineLearning project/project/'

# storing image id and images components of training images in dictonaries
training_image_dictionary = {}
# reading training images from train folder
for filename in glob.glob(train_image_folder + '*.jpg'):

    try:

        training_image_dictionary[os.path.splitext(os.path.basename(filename))[0]] = cv2.imread(
            os.path.normpath(filename))

        try:  # storing b,g,r and gray scale images in numpy array and resizing colour into 100 * 100 and grayscale in 200*200
            training_image_dictionary[os.path.splitext(os.path.basename(filename))[0]] = cv2.split(
                cv2.resize(training_image_dictionary[os.path.splitext(os.path.basename(filename))[0]],
                           dsize=(100, 100))), cv2.resize(cv2.imread(os.path.normpath(filename), 0), dsize=(200, 200))



        except:  # printing the error files
            print filename
            if (training_image_dictionary.__contains__(os.path.splitext(os.path.basename(filename))[0])):
                training_image_dictionary.__delitem__(os.path.splitext(os.path.basename(filename))[0])
                pass

    except:
        print filename
        pass

print 'succesful image extraction of training folder'
# storing image id and images components of testing images in dictonaries
testing_image_dictionary = {}
# list for storing iids of invalid images
testing_error_images = []
for filename in glob.glob(test_image_folder + '*.jpg'):

    try:
        testing_image_dictionary[os.path.splitext(os.path.basename(filename))[0]] = cv2.imread(
            os.path.normpath(filename))
        try:  # storing b,g,r and gray scale images in numpy array and resizing colour into 100 * 100 and grayscale in 200*200
            testing_image_dictionary[os.path.splitext(os.path.basename(filename))[0]] = cv2.split(
                cv2.resize(testing_image_dictionary[os.path.splitext(os.path.basename(filename))[0]],
                           dsize=(100, 100))), cv2.resize(cv2.imread(os.path.normpath(filename), 0), dsize=(200, 200))
        except:  # printing the error files
            testing_error_images.append(os.path.splitext(os.path.basename(filename))[0])
            print filename
            if (testing_image_dictionary.__contains__(os.path.splitext(os.path.basename(filename))[0])):
                testing_image_dictionary.__delitem__(os.path.splitext(os.path.basename(filename))[0])
            pass


    except:
        print filename
        pass

print 'succesful image extraction of testimages'
# read train csv
f = open(train_csv_folder + 'train.csv')
csv_f = csv.reader(f)
# classes and posterior probability matrix interface
csvdecison_dictionary = {'10000000': 1, '01000000': 2, '00100000': 3, '00010000': 4, '00001000': 5, '00000100': 6,
                         '00000010': 7, '00000001': 8}
class_label = {}
f.next()
for row in csv_f:  # converting posterior probabilties into decsion classes
    class_label[row[0]] = csvdecison_dictionary[(','.join(row[1:]).replace(',', ''))]
# storing decision classes in the order of keys sored in dictionary
y_list_train = []
for very in training_image_dictionary.keys():
    y_list_train.append(float(class_label[very]))
# creating a ndarray
y_list_train = np.array(y_list_train)
# storing intensities of grayscale images
train_gray_matrix = map(lambda x: x[1], training_image_dictionary.values())
test_gray_matrix = map(lambda x: x[1], testing_image_dictionary.values())

# storing b,g,r, arrays in a list of all images of training folder
train_b_matrix = map(lambda x: x[0][0], training_image_dictionary.values())
train_g_matrix = map(lambda x: x[0][1], training_image_dictionary.values())
train_r_matrix = map(lambda x: x[0][2], training_image_dictionary.values())

# halving the image horizontally and calculating the intensities of red ,,green blue dimensions of the training image
blue_train_top = map(lambda x: [np.mean(x[0:51, :])], train_b_matrix)
blue_train_bottom = map(lambda x: [np.mean(x[51:, :])], train_b_matrix)
green_train_top = map(lambda x: [np.mean(x[0:51, :])], train_g_matrix)
green_train_bottom = map(lambda x: [np.mean(x[51:, :])], train_g_matrix)
red_train_top = map(lambda x: [np.mean(x[0:51, :])], train_r_matrix)
red_train_bottom = map(lambda x: [np.mean(x[51:, :])], train_r_matrix)

# storing b,g,r, arrays in a list of all images of testing folder
test_b_matrix = map(lambda x: x[0][0], testing_image_dictionary.values())
test_g_matrix = map(lambda x: x[0][1], testing_image_dictionary.values())
test_r_matrix = map(lambda x: x[0][2], testing_image_dictionary.values())

# halving the image horizontally and calculating the intensities of red ,,green blue dimensions of the training image
blue_test_top = map(lambda x: [np.mean(x[0:51, :])], test_b_matrix)
blue_test_bottom = map(lambda x: [np.mean(x[51:, :])], test_b_matrix)
green_test_top = map(lambda x: [np.mean(x[0:51, :])], test_g_matrix)
green_test_bottom = map(lambda x: [np.mean(x[51:, :])], test_g_matrix)
red_test_top = map(lambda x: [np.mean(x[0:51, :])], test_r_matrix)
red_test_bottom = map(lambda x: [np.mean(x[51:, :])], test_r_matrix)

# creating list of histograms for blue,red,green dimensions for both training and testing
hist_b_list = map(lambda x: cv2.calcHist([x], [0], None, [256], [0, 256]).flatten(), train_b_matrix)
hist_g_list = map(lambda x: cv2.calcHist([x], [0], None, [256], [0, 256]).flatten(), train_g_matrix)
hist_r_list = map(lambda x: cv2.calcHist([x], [0], None, [256], [0, 256]).flatten(), train_r_matrix)

hist_test_b_list = map(lambda x: cv2.calcHist([x], [0], None, [256], [0, 256]).flatten(), test_b_matrix)
hist_test_g_list = map(lambda x: cv2.calcHist([x], [0], None, [256], [0, 256]).flatten(), test_g_matrix)
hist_test_r_list = map(lambda x: cv2.calcHist([x], [0], None, [256], [0, 256]).flatten(), test_r_matrix)

# getting the ids of test images in dictionary
y_test_id = []
for each in testing_image_dictionary.keys():
    y_test_id.append([each])

# conconateing the different colour momemnts of both testing and training
colour_moments_train = np.concatenate(
    (blue_train_top, blue_train_bottom, green_train_top, green_train_bottom, red_train_top, red_train_bottom), axis=1)

colour_moments_test = np.concatenate(
    (blue_test_top, blue_test_bottom, green_test_top, green_test_bottom, red_test_top, red_test_bottom), axis=1)

# calculating hog features for both test and train
hog_train_features = map(lambda x: hog(x, orientations=8, pixels_per_cell=(20, 20), cells_per_block=(1, 1)),
                         train_gray_matrix)
hog_test_features = map(lambda x: hog(x, orientations=8, pixels_per_cell=(20, 20), cells_per_block=(1, 1)),
                        test_gray_matrix)

# applying pca on histograms of 3 colours on both testing and training images
pca = decomposition.PCA(10)
pca.fit(np.array(hist_b_list))
hist_b_pca = pca.transform(np.array(hist_b_list))
hist_test_b_pca = pca.transform(np.array(hist_test_b_list))
pca.fit(np.array(hist_r_list))
hist_r_pca = pca.transform(np.array(hist_r_list))
hist_test_r_pca = pca.transform(np.array(hist_test_r_list))
pca.fit(np.array(hist_g_list))
hist_g_pca = pca.transform(np.array(hist_g_list))
hist_test_g_pca = pca.transform(np.array(hist_test_g_list))

# conconatening the features  of training
rg_pca = np.hstack((hist_r_pca, hist_g_pca))
rgb_pca = np.hstack((rg_pca, hist_b_pca))
col_hist = np.hstack((colour_moments_train, rgb_pca))
features = np.hstack((np.array(hog_train_features), col_hist))

# conconatening the features of testing
rg_pca_test = np.hstack((hist_test_r_pca, hist_test_g_pca))
rgb_pca_test = np.hstack((rg_pca_test, hist_test_b_pca))
col_hist_test = np.hstack((colour_moments_test, rgb_pca_test))
features_test = np.hstack((np.array(hog_test_features), col_hist_test))

# intializing the logistic regression
lin_clf = linear_model.LogisticRegression()


kfold = range(1,20)
idx =1
#validation accuracy holder
validation_logloss_scores =[]
for e in  kfold:
    X_train, X_validate, Y_train, Y_validate= train_test_split( features, y_list_train, test_size=0.10 )



    # training the logistic regression classifeir with training data
    lin_clf.fit(X_train, Y_train)    # calculating validation accuracy
    score_validation = lin_clf.score(X_validate, Y_validate)
    print 'accuracy in percentage on validation data '
    print  idx
    # printing validation score
    print score_validation
    validation_logloss_scores.append(score_validation)
    # posterio probabilites of validaation dataset
    proba = lin_clf.predict_proba(X_validate)

    # posterio probabilites of testing dataset
    test_proba = lin_clf.predict_proba(features_test)
    # calculating logloss score of validation data
    logloss_score = log_loss(Y_validate, proba)
    print 'validation log loss score'
    print logloss_score
    validation_logloss_scores.append(logloss_score)

    error_images = []

    # applying equal probabilites for invalid images
    for every in testing_error_images:
        error_images.append([every, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    # appending invalid images and their posterior probabilities,writing the posterior probabilites into csv file
    posterior_probabilites_test_data = np.vstack((np.array(error_images), np.hstack((np.array(y_test_id), test_proba))))
    np.savetxt("output" + str(idx) + ".csv", posterior_probabilites_test_data.astype(dtype=float), delimiter=",",
               fmt='%1.5f', header='id,col1,col2,col3,col4,col5,col6,col7,col8')
    idx += 1

# selecting the best score depending upon validation data
selected_file= validation_logloss_scores.index(max(validation_logloss_scores)) + 1
print 'the best output is in'+" output" + str(selected_file) + ".csv" +' best score is '+str(max(validation_logloss_scores))