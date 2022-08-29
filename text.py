import cv2
import numpy as np
import time
import sys
import os.path
import random

def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img


def filter_matches(matches, ratio):
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    return good


def filter_radius_matches(matches, ratio):
    good = []
    for match in matches:
        if len(match) >=2:
            m = match[0]
            n = match[1]
            if m.distance < ratio*n.distance:
                good.append(m)
    return good
def pad_numbers(num):
    str_num = str(num)
    num_length = 4-len(str_num)
    #print(num)
    for i in range(num_length):
        #print(i)
        str_num = "0" + str_num
    #print(str_num)
    return str(str_num)
def batch_of_detectors(feature_det, feature_det_name):
    header = "feature_detector,file1_number,file2_number,ratio,detect_time,merge_time,total_time\n"
    out_csv = ""
    for i in range (1, 20):
        for j in range(1,3):
            for k in range(1,10):
                ratio = k/10
                file1= "images\\original\\image_" + pad_numbers(i) + ".jpg"
                file2= "images\\original\\image_" + pad_numbers(i+j) + ".jpg"     
                print("======================")
                print(file1)
                print(file2)
                print("ratio:" + str(ratio))
                print("======================")

                output_dir = "output\\" + feature_det_name + "\\" + pad_numbers(i) + "_" + pad_numbers(i+j) + "_" + str(ratio) + ".jpg"
                out_arr = create_MergedImage(file1,file2,ratio,output_dir, feature_det, feature_det_name)
                if (out_arr) is not None:
                    out_csv += write_csv_array(out_arr)
    f = open("results_"+ feature_det_name +".csv", "w")
    f.write(header+out_csv)
def sift_batch():
    num_features = 10
    #while num_features < 400:
    contrast_threshold = 10

    while contrast_threshold > 0:
        feature_det = cv2.SIFT_create(nfeatures=30, edgeThreshold=10,contrastThreshold=0.2)
        create_MergedImage(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0021.jpg", r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0022.jpg", 0.8    , "output\\tests\\sift_021_022_nfeatures_" +str(num_features) + "_sigma_threshold_" +str(contrast_threshold) + ".jpg", feature_det, "det_name")
        contrast_threshold -= 1
    #num_features += 1
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
def write_csv(feature_detector, file1, file2, ratio, total_time, merge_time, detect_time):
    out_file = "Results.csv"
    file1_number = file1.split("_")[-1]
    file2_number = file2.split("_")[-1]
    csv_line = str(feature_detector) +"," +str(file1_number) + "," + str(file2_number) + "," + str(ratio) + "," + str(detect_time) + "," + str(merge_time) + ","+ str(total_time) + "\n"
    return csv_line
def write_csv_array(input_arr):
    print(input_arr)
    out_file = "Results.csv"
    file1_number = input_arr[1].split("_")[-1]
    file2_number = input_arr[2].split("_")[-1]
    csv_line = str(input_arr[0]) +"," +str(file1_number) + "," + str(file2_number) + "," + str(input_arr[3]) + "," + str(input_arr[6]) + "," + str(input_arr[5]) + ","+ str(input_arr[4]) + "\n"
    return csv_line

    #header = "feature_detector,file1_number,file2_number,ratio,detect_time,merge_time,total_time\n"
def create_MergedImage(file1, file2, ratio, output_file, feature_detector, det_name, contrast_threshold=0.09, nOctaveLayer=3, edge_threshold=1):

    img_ = cv2.imread(file1)

    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

    img = cv2.imread(file2)

    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    detect_start = time.time()
    start_time = time.time()

    kp1, des1 = feature_detector.detectAndCompute(img1,None)
    # find key points
    kp2, des2 = feature_detector.detectAndCompute(img2,None)


    match = cv2.BFMatcher()
    matches = match.knnMatch(des1,des1,k=2)
    #matches = match.match(des1,des2)
    #matches = match.radiusMatch(des1,des1,100.0)
    good = filter_matches(matches, ratio)
    detect_time = time.time()-detect_start
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        cv2.namedWindow("original_image_overlapping.jpg", cv2.WINDOW_NORMAL)
        cv2.imwrite("original_image_overlapping.jpg", img2)
        cv2.destroyAllWindows()
        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.namedWindow("matching_points.jpg", cv2.WINDOW_NORMAL)
        cv2.imshow("matching_points.jpg", img3)
        cv2.waitKey(0)
    else:
        print("Not enough matches are found - %d/%d", len(good),MIN_MATCH_COUNT)
        return
    print("test")
    second_time_start = time.time()
    dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))


    #dst = warpImages(img, img_,M)
    dst[0:img.shape[0],0:img.shape[1]] = img
    total_time = time.time() -start_time
    
    second_time = time.time() - second_time_start
    #print(det_name)
    #print("Merge Time: "+ str(second_time))
    #print("Total Time: " + str(total_time))
    #print("Detect Time: " + str(detect_time))
    cv2.imwrite(output_file + det_name + ".jpg", trim(dst))
    return [det_name, file1, file2, ratio, total_time, second_time, detect_time]
def create_MergedImage2():
    height = 0
    width = 0
    #,cv2.IMREAD_GRAYSCALE
    img1 = cv2.imread(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0001.jpg") # queryImage
    #img2 = cv2.imread(r"C:\Users\mattp\Pictures\wheel_2.PNG")
    #img1 = img1[0:img1.shape[0],400:img1.shape[1]]
    #cv2.namedWindow("cropped.jpg", cv2.WINDOW_NORMAL)
    # cv2.imshow("cropped.jpg", trim(img1))

    img2 = cv2.imread(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0003.jpg") # trainImage
    #img1 = cv2.imread(r"C:\Users\mattp\Pictures\wheel_1.PNG")

    #img2 = img2[img2.shape[0]/2:img2.shape[0]]

    # Initiate SIFT detector
    #sift = cv2.SIFT_create(contrastThreshold=0.1)
    sift = cv2.SIFT_create(contrastThreshold=0.09, edgeThreshold=10)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    MIN_MATCH_COUNT = 10

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    dst = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
    #dst[0:img2.shape[0],0:img2.shape[1]] = img2

    #cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.namedWindow("matching_points.jpg", cv2.WINDOW_NORMAL)
    cv2.imshow("matching_points.jpg", img3)

    #plt.imshow(img3),plt.show()

    cv2.waitKey(0)


def create_MergedImageRandomized():

    height = 0
    width = 0
    #,cv2.IMREAD_GRAYSCALE
    img1 = cv2.imread(r"C:\Users\mattp\Downloads\frame1.jpg") # queryImage
    img2 = cv2.imread(r"C:\Users\mattp\Downloads\frame2.jpg") # trainImage
    img2Holder = img2

    #img2 = cv2.imread(r"C:\Users\mattp\Pictures\wheel_2.PNG")
    #img1 = img1[0:img1.shape[0],400:img1.shape[1]]
    #cv2.namedWindow("cropped.jpg", cv2.WINDOW_NORMAL)
    # cv2.imshow("cropped.jpg", trim(img1))
    startX = 0
    startY = int(img1.shape[1] / 2 - 1)
    #  image2 = img[0:-1,img.shape[1]-200:-1]
    cols = int(img1.shape[0])
    rows = int(img1.shape[1])

    img2 = img2[0:-1,startY:cols]

    #img1 = img1[0:-1,startY:-1];
    #cv2.namedWindow("Hellooooo", cv2.WINDOW_NORMAL)

    #cv2.imshow("Hellooooo",img2)

    #img1 = cv2.imread(r"C:\Users\mattp\Pictures\wheel_1.PNG")

    #img2 = img2[img2.shape[0]/2:img2.shape[0]]

    # Initiate SIFT detector
    #sift = cv2.SIFT_create(contrastThreshold=0.1)
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    #kp1 = sift.detect(img1,None)
    kp1, des1 = sift.detectAndCompute(img1,None)


   #des1 = sift.compute(img1,kp1)
    #Find Keypoints
    kp2 = sift.detect(img2)

    for kp in kp2:
        kp.pt = (kp.pt[0] + startY,kp.pt[1] + startX)
    img2 = img2Holder
    des2 = sift.compute(img2, kp2)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2[1],k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    MIN_MATCH_COUNT = 10
    print(len(good))
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    dst = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
    #dst[0:img2.shape[0],0:img2.shape[1]] = img2

    #cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.namedWindow("matching_points.jpg", cv2.WINDOW_NORMAL)
    cv2.imwrite("matching_points.jpg", img3)

    cv2.imshow("matching_points.jpg", dst)

    #plt.imshow(img3),plt.show()

    cv2.waitKey(0)


def stitchImages():
    img_ = cv2.imread(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0001.jpg")
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img = cv2.imread(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0003.jpg")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2Holder = img2

    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
    matches = np.asarray(good)


    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    #print H
    else:
        raise AssertionError("Can't find enough keypoints.")
    dst = cv2.warpPerspective(img_, H, (img.shape[1] + img_.shape[1], img.shape[0]))

    dst[0:img.shape[0], 0:img.shape[1]] = img
    cv2.imwrite('output.jpg', dst)

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
        cols = int(img1.shape[0] * alpha)
        rows = int(img1.shape[1] * alpha)

        startX = random.randint(0,img1.shape[0] - cols)
        startY =  random.randint(0,img1.shape[1] - rows)
        #startY = int(img1.shape[1] / 2 - 1)
        print(startY)
        img2Holder = img2
        img2 = img2[0:-1,startY:startY + cols]
        #cv2.imshow("test", img2)
       # cv2.waitKey(0)
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
        match_time = startTime = time.time()

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

#ratio = float(sys.argv[1])
#file1 = sys.argv[2]
#file2 = sys.argv[3]
#output_dir = sys.argv[4]

#create_MergedImage(file1,file2,ratio,output_dir, cv2.SIFT_create(), "SIFT")

#create_MergedImage(file1,file2,ratio,output_dir, cv2.ORB_create(), "ORB")

#create_MergedImage(file1,file2,ratio,output_dir, cv2.AKAZE_create(), "AKAZE")

#create_MergedImage(file1,file2,ratio,output_dir, cv2.KAZE_create(), "KAZE")

#batch_of_detectors(cv2.SIFT_create(), "SIFT")
#batch_of_detectors(cv2.ORB_create(), "ORB")
#batch_of_detectors(cv2.AKAZE_create(), "AKAZE")
#batch_of_detectors(cv2.KAZE_create(), "KAZE")

#batch_of_detectors(cv2.FastFeatureDetector_create(), "FAST")
#batch_of_detectors(cv2.GFTTDetector_create(), "GFTT")
#batch_of_detectors(cv2.BRISK_create(), "BRIEF")
#batch_of_detectors(cv2.MSER_create(), "MSER")
#sift_batch()
#feature_det = cv2.SIFT_create()
#create_MergedImage(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0021.jpg",
#                   r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0022.jpg", 0.75,
#                   "testing.jpg", feature_det, "det_name")
#create_MergedImage2()
stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
img2 = cv2.imread(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0263.jpg")  # trainImage
img1 = cv2.imread(r"C:\Users\mattp\Documents\thesis_Stuff\images\original\image_0266.jpg")  # trainImage

#foo = cv2.imread("D:/foo.png")
#bar = cv2.imread("D:/bar.png")

#create_MergedImageRandomized()
#stitchImagesRandomized(0.5)
result = stitcher.stitch((img1,img2))
#cv2.namedWindow("D:/result.jpg", cv2.WINDOW_NORMAL)
#print(result)
cv2.imshow("D:/result.jpg", result[1])
cv2.waitKey(0)
