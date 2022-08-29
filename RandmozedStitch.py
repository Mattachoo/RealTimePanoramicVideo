
import cv2
import numpy as np
import time
import sys
import os.path
import random


def stitchImagesRandomized(alpha):
    result = False
    runs = 0
    out_csv = "run,alpha,img1_detect,img1_compute,img1_keypoints,img2_detect,img_2_compute,img2_keypoints,keypoint_shift,copy,match_time,matches_found,filter,stitch_time,total_time\n"
    while not result:
        runs +=1
        print(runs)
        #Stitching code taken from https://towardsdatascience.com/image-stitching-using-opencv-817779c86a83
        #Modified to randomization
        img_ = cv2.imread(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0001.jpg")
        img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        img = cv2.imread(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0003.jpg")
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT

        #  image2 = img[0:-1,img.shape[1]-200:-1]
        cols = int(img2.shape[0] * alpha)
        rows = int(img2.shape[1] * alpha)

        startX = random.randint(0,img2.shape[0] - cols)
        #startY =  random.randint(0,img2.shape[1] - rows)
        startY = int(img1.shape[1] / 2 - 1)
        print("StartY: " + str(startY) + ", cols: " + str(cols))
        img2Holder = img2
        img2 = img2[0:-1,startY:startY + cols]
        print("Img2 size: " + str(img2.shape[0])+", "+ str(img2.shape[1]))

        cv2.imshow("test", img2)
        cv2.waitKey(0)
        startTime = time.time()
        kp1 = sift.detect(img1)
        img1_detect = (time.time() - startTime)
        startTime = time.time()

        kp1, des1 = sift.compute(img1,kp1)
        img1_compute = (time.time() - startTime)

        startTime = time.time()
        kp2 = sift.detect(img2)
        img2_detect = (time.time() - startTime)
        startTime = time.time()

        for kp in kp2:
            kp.pt = (kp.pt[0] + startY,kp.pt[1] + startX)
        keypoint_shift = (time.time() - startTime)

        startTime = time.time()
        img2 = img2Holder
        copy_time = (time.time() - startTime)

        startTime = time.time()

        kp2, des2 = sift.compute(img2, kp2)
        imgs2_compute = (time.time() - startTime)
        #kp2, des2 = sift.detectAndCompute(img2,None)

        bf = cv2.BFMatcher()
        startTime = time.time()
        matches = bf.knnMatch(des1,des2, k=2)
        print(len(matches))
        match_time = startTime - time.time()

        # Apply ratio test
        startTime = time.time()
        good = []
        for m in matches:
            if m[0].distance < 0.5 * m[1].distance:
                good.append(m)
        matches = np.asarray(good)


        if len(matches[:,0]) >= 4:
            result = True
            src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        else:
            filter_time = (time.time() - startTime)

            #update with process info so far here
            #    out_csv = "run,alpha,img1_detect,img1_compute,img1_keypoints,img2_detect,img_2_compute,img2_keypoints,keypoint_shift,copy,match_time,matches_found,filter,stitch_time, total_time"
            out_csv+= str(runs) + "," +  str(alpha) + "," + str(img1_detect) + "," + str(img1_compute) + "," + str(len(kp1)) + "," + str(img2_detect) + "," + str(imgs2_compute) + "," + str(len(kp2)) + "," + str(keypoint_shift) + "," + str(copy_time) + "," + str(match_time) + "," + str(len(matches)) + "," + str(filter_time) + "\n"
            continue
            #raise AssertionError("Can't find enough keypoints.")
        filter_time = (time.time() - startTime)

        full_keypoints = cv2.drawKeypoints(img_,kp1,0, (0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('output_fullKeypoints.jpg', full_keypoints)

        small_keypoints = cv2.drawKeypoints(img,kp2,0, (0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('output_smallKeypoints.jpg', small_keypoints)

        img3 = cv2.drawMatchesKnn(img, kp1, img_, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        startTime = (time.time())
        dst = cv2.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))

        dst[0:img.shape[0], 0:img.shape[1]] = img
        stitch_time = (time.time() - startTime)
        cv2.imwrite('output_randomized.jpg', dst)
        cv2.imwrite("Matching_keypoints.jpg",img3)
        #out_csv = "run,alpha,img1_detect,img1_compute,img1_keypoints,img2_detect,img_2_compute,img2_keypoints,keypoint_shift,copy,match_time,matches_found,filter,stitch_time, total_time"
        sum_time = img1_detect + img1_compute + img2_detect + imgs2_compute + keypoint_shift + copy_time + match_time + filter_time + stitch_time
        out_csv += str(runs) + "," + str(alpha) + "," + str(img1_detect) + "," + str(img1_compute) + "," + str(
            len(kp1)) + "," + str(img2_detect) + "," + str(imgs2_compute) + "," + str(len(kp2)) + "," + str(
            keypoint_shift) + "," + str(copy_time) + "," + str(match_time) + "," + str(len(matches)) + "," + str(
            filter_time) + "," + str(stitch_time) +","  +str(sum_time)+"\n"
    with open('output.csv', 'w') as outfile:
        outfile.write(out_csv)

stitchImagesRandomized(0.5)