import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

import os
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


def generate_dataset(true_data_dirs, false_data_dir):
    data = []
    targets = []
    print("Reading Data")
    count = 0
    count2 = 0
    cap = 100
    step = 0
    skip = 1000
    for true_data_dir in true_data_dirs:
        step = 0
        count = 0

        for item in os.listdir(true_data_dir):
            if(step < skip):
                step+=1
                continue
            file = os.path.join(true_data_dir, item)
            image = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (600,400)).flatten()
            #print(type(image))
            data.append(image)
            targets.append(2)
            count += 1
            if count >= cap/len(true_data_dirs):
                break
            #print(len(image))
            #print(type(image))
        print("Done Reading Truths")
    for item in os.listdir(false_data_dir):
        file = os.path.join(false_data_dir, item)
        image = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (600,400)).flatten()
        data.append(image)
        targets.append(1)
        count2 +=1

        if count2 >= count:
            break
    print("Done Reading Data")
    header = ['data', 'target']
    #df = pd.DataFrame([data, targets], columns=header)
    return train_test_split(data, targets, test_size=0.8)


def generate_dataset_filenames(true_data_dirs, false_data_dir, sample_size):
    data = []
    targets = []
    print("Reading Data")
    count = 0
    count2 = 0
    cap = 1000000000000
    step = 0
    skip = 0
    for true_data_dir in true_data_dirs:
        step = 0
        count = 0

        for item in os.listdir(true_data_dir):
            if(step < skip):
                step+=1
                continue
            file = os.path.join(true_data_dir, item)
            #image = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (600,400)).flatten()
            #print(type(image))
            data.append(file)
            targets.append(1)
            count += 1
            if count >= cap/len(true_data_dirs):
                break
            #print(len(image))
            #print(type(image))
        print("Done Reading Truths")
    for item in os.listdir(false_data_dir):
        file = os.path.join(false_data_dir, item)
        #image = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (600,400)).flatten()
        data.append(file)
        targets.append(0)
        count2 +=1

        if count2 >= count:
            break
    print("Done Reading Data")
    header = ['data', 'target']
    df = pd.DataFrame()
    df['file_name'] = data
    df['target'] = targets
    sample_df = df.sample(sample_size)
    data = []
    for filename in sample_df['file_name']:
        #print(filename)
        data.append(cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (600,400)).flatten())
    #data = pd.Series(X_train_pic)


   # print(sample_df.head(5))
    #print(sample_df['target'])
    #data_2 = sample_df[:,:-1]
    #target_2 = sample_df.iloc[:, -1]

    return train_test_split(data,sample_df['target'], test_size=0.8)


def logreg(X_train, X_test, y_train, y_test):
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    logit = LinearRegression()
    logit.fit(X_train, y_train)
    #print("Loss:", logit.lo)
    print(logit.score(X_test, y_test))
    #print(logit.predict(X_test))
    startTime = time.time()
    #print(X_test[1])
    #pred_prob = logit.predict_proba(X_test)
    pred = logit.predict(X_test)
    test = pd.DataFrame()
    test['pred'] = pred
    test['target'] = y_test
    #test['predict_proba'] = pred_prob[:,1]
    print(test.loc[test['pred'] > 1])
    #print(time.time() - startTime)
    #for x in logit.predict_proba(X_test):
    #    print(x)
    return logit


def logreg_filename(X_train, X_test, y_train, y_test):
    X_train_pic = []
    for filename in X_train:
        #print(filename)
        X_train_pic.append(cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (600,400)).flatten())
    X_train_pic = pd.Series(X_train_pic)

    X_test_pic = []
    for filename in X_test:
        #print(filename)
        X_test_pic.append(cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (600,400)).flatten())
    y_train_pic = pd.Series(X_test_pic)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train_pic)
    X_test = ss.transform(X_test_pic)
    logit = LogisticRegression()
    logit.fit(X_train, y_train_pic)
    print(logit.score(X_test, y_test))

def cross_score(model, model_name, n_splits, X, y):

    ### your code here ###
    cross_val = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cross_val_scores = cross_val_score(model, X, y, cv=cross_val)
    avg_score = round(cross_val_scores.sum() / len(cross_val_scores), 3)
    scores_df = pd.DataFrame(cross_val_scores)
    std_dev = round(scores_df.std()[0], 3)
    print(model_name, "with", n_splits, "splits has an average score of", avg_score, "+/-", std_dev)


def keras_test(X_train, X_test, y_train, y_test):
    # create a feedforward model
    model = Sequential()
    input_size = len(X_train[0])
    # model.add(Dense(17, input_shape=(input_size,), init='uniform',activation='relu'))
    model.add(Dense((input_size * 2)+1, input_dim=input_size, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test_c), epochs=30, batch_size=20, verbose=0);
    score =model.evaluate(X_test, y_test, batch_size=20,  verbose=0)
    print("Keras:", score)


def main():
#    X_train, X_test, y_train, y_test = generate_dataset_filenames(["videos_og","videos_og_1"],"videos_shifted",1000)
#    logreg_filename(X_train, X_test, y_train, y_test)
    #X_train, X_test, y_train, y_test = generate_dataset(["videos_og","videos_og_1"],"videos_shifted")
    X_train, X_test, y_train, y_test = generate_dataset_filenames(["videos_og_1"],"videos_shifted",1000)

    logreg(X_train, X_test, y_train, y_test)
    #keras_test(X_train, X_test, y_train, y_test)
    name = "Logistic Regression"
    print("starting")

def get_model(videos_true, videos_false, sample_size):
    #X_train, X_test, y_train, y_test = generate_dataset_filenames(["videos_og","videos_og_1"],"videos_shifted",1000)
    X_train, X_test, y_train, y_test = generate_dataset_filenames(videos_true,videos_false,sample_size)

    return logreg(X_train, X_test, y_train, y_test)
if __name__ == "__main__":
    main()
