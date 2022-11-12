import pandas as pd
import sklearn.metrics
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
import sklearn
from datetime import datetime
import os
import pickle
#df = pd.read_csv('label_filtered.csv')
#print(df["seam"].value_counts())
def convert_images_to_df(file_path, num_of_entries,label):
    image_data = []
    label_data = []
    count = 0
    for file in os.listdir(file_path):
            label_data.append(label)
            #            file = os.path.join(true_data_dir, item)

            image_path = os.path.join(file_path, file)
            img = imread(image_path)
            #resize image so all images are the same size
            img = resize(img,(300,300,3))

            image_data.append(img.flatten())
            count += 1
            if(count > num_of_entries and num_of_entries > 0):
                break
            print(".", end="")
    image_data = np.array(image_data)
    label_data = np.array(label_data)

    df = (pd.DataFrame(image_data))
    df['seam'] = (label_data)
    #print(df.head(5))
    return df
    #df.to_csv("df_output.csv")


def train_model(df, kernal, coef0, gamma):

    x = df.iloc[:, :-1]  # input data
    y=df.iloc[:,-1] #output data
    svc = svm.SVC(probability=True,kernel=kernal,coef0=coef0,gamma=gamma)
    print("Splitting")
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2,random_state=77,stratify=y)
    print("Fitting")
    svc.fit(x_train,y_train)

    y_pred = svc.predict(x_test)

    accuracy_score = sklearn.metrics.accuracy_score(y_pred,y_test)
    print("Accuracy: ",accuracy_score)
    return svc

def train_model_2(df):

    x = df.iloc[:, :-1]  # input data
    y = df.iloc[:, -1]  # output data
    svc = svm.SVC(probability=True)
    print("Splitting")
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=77,
                                                                                stratify=y)
    print("Fitting")
    svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)

    accuracy_score = sklearn.metrics.accuracy_score(y_pred, y_test)
    print("Accuracy: ", accuracy_score)
    return svc
    #Code from "https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01", useful to test later
    #param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
    #model=GridSearchCV(svc,param_grid)


def score_model(svc, directory):
    scores = dict()
    for root, subdirectories, files in os.walk(directory):

        for file in files:
            print(file)
            if ".jpg" in file:
                path = (os.path.join(root, file))
                #print(path)
                tool = root.replace("output\\", "")
                if tool not in scores.keys():
                    scores[tool] = [0, 0]

                img = imread(path)
                # resize image so all images are the same size
                img = resize(img, (300, 300, 3))
                image_data = [img.flatten()]
                #print(svc.predict(image_data)[0])
                if int(svc.predict(image_data)[0].replace("\n", "")):
                    scores[tool][0] += 1
                scores[tool][1] += 1
    with open(directory + "scores.csv", "a") as out_map:
        for key in scores.keys():
            print(key + "," + str(scores[key][0]) + "," + str(scores[key][1]) + "\n")
            out_map.write(key + "," + str(scores[key][0]) + "," + str(scores[key][1]) + "\n")


def parameter_sweep(df):
    kernals = {'linear', 'poly', 'rbf', 'sigmoid'}
    gammas = {'scale', 'auto'}
    score = 0
    scores = "kernnal,coef0,gamma"
    #do something for coef0
    for kernal in kernals:
        coef0 = 0.0
        for gamma in gammas:
            if kernal == "poly" or kernal == "sigmoid":
                while coef0 <= 1:
                    print(kernal + "," + str(coef0) + "," + gamma+ ","+str(score))

                    score = train_model(df,kernal,coef0,gamma)
                    scores += kernal + "," + str(coef0)+"," + gamma+ ","+str(score) +"\n"
                    coef0 += 0.1
            else:
                print(kernal + "," + str(coef0) + "," + gamma+ ","+str(score))

                score = train_model(df, kernal, coef0, gamma)
                scores += kernal + "," + str(coef0) + "," + gamma + ","+str(score) + "\n"
    with open("../../Documents/thesis_Stuff/Train_out.csv", "w") as out_file:
        out_file.write(scores)
def train():
    print("Reading")
    dt = datetime.now()
    dataframe = (convert_images_to_df("videos_og", 500,1))

    print(dataframe.shape)
    dataframe = pd.concat([dataframe,convert_images_to_df("videos_og_1", 500,1)],axis=0)
    print(dataframe.shape)

    dataframe = pd.concat([dataframe,convert_images_to_df("videos_shifted", 1000,0)], axis=0)
    print(dataframe.shape)

    ts = datetime.timestamp(dt)
    print("Timestamp is:", ts)
    #df = pd.read_csv('df_output.csv')
    print("Reading done")
    svc = train_model(dataframe,"rbf",0.0,"scale")
    with open("../../Documents/thesis_Stuff/model.yml", "wb") as file:
        pickle.dump(svc, file)
    return svc
    #parameter_sweep(dataframe)


svc = train()
directory = "output"
dirs = [r"C:\Users\mattp\Documents\thesis_Stuff\RandomizedStitch\1-1",r"C:\Users\mattp\Documents\thesis_Stuff\RandomizedStitch\1-3"]
for directory in dirs:
    score_model(svc, directory)

#with open("model.yml", "rb") as file:
    #svc = pickle.load(file)
    #irectory = r"C:\Users\mattp\Documents\thesis_Stuff\RandomizedStitch\1-1"
    #score_model(svc, directory)
