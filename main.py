import math
import random
import time

import cv2
import numpy as np


def findInitialHomography():
    pass


def mergeImges():
    pass

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
def fixHMatrixStitch(frames, H):
    img1 = frames[1]
    img2 = frames[0]

    dst = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img1.shape[0]))
    shift = 0
    dst[0:img2.shape[0], 0+shift:img2.shape[1]+shift    ] = img2
    return dst
def partitionKeypoints(img1, kp1, des1):
    regions = [[],[],[],[]]
    out_kp = []
    out_des = []
    kps_ = [[],[],[],[]]
    des_ = [[],[],[],[]]
    for i in range(0, len(kp1)):
        x = kp1[i].pt[0]
        y = kp1[i].pt[1]
        mid_x = int(img1.shape[1]/2)
        mid_y = int(img1.shape[0]/2)
        #Top left
        if x < mid_x:
            #Top Left
            if y < mid_y:
                kps_[0].append(kp1[i])
                des_[0].append(des1[i])

                #regions[0].append([kp1[i], des1[i]])
            #Bottom Left
            else:
                kps_[2].append(kp1[i])
                des_[2].append(des1[i])
                #regions[2].append([kp1[i], des1[i]])
        else:
            #Top Right
            if y < mid_y:
                kps_[1].append(kp1[i])
                des_[1].append(des1[i])
                #regions[1].append([kp1[i], des1[i]])
            #Bottom Right
            else:
                kps_[3].append(kp1[i])
                des_[3].append(des1[i])
                #regions[3].append([kp1[i], des1[i]])
    count = 0

# region in regions:
    #    region = [kps_, des_]
    #    kps = []
    #    for kp in region:
    #        kps.append(kp[0])
    #    print(len(kps))
    #    output_imaage = cv2.drawKeypoints(img1, kps,0,(0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    #    cv2.imwrite("regional_kp_" + str(count) + ".jpg",output_imaage)
    #    region[0] = kps
    #   count+=1
    for kp in kps_:
       #print(len(kp))
        output_imaage = cv2.drawKeypoints(img1, kp,0,(0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("regional_kp_" + str(count) + ".jpg",output_imaage)
        count += 1
    return [kps_, des_]
def partitionKeypoints2(img1, kp1, des1, boundries):
    #regions = [[]] * len(boundries)
    out_kp = []
    out_des = []

    kps_ =  []
    des_ =[]

    for boundry in boundries:
        kps_.append([])
        des_.append([])
    for i in range(0, len(kp1)):
        x = kp1[i].pt[0]
        y = kp1[i].pt[1]
        #Below section needs to be updated to use given regions
        for region_boundry_index in range(0, len(boundries)):

            x_min = boundries[region_boundry_index][0]
            x_max = boundries[region_boundry_index][1]
            y_min = boundries[region_boundry_index][2]
            y_max = boundries[region_boundry_index][3]
            #mid_x = int(img1.shape[1]/2)
            #mid_y = int(img1.shape[0]/2)
            #Top left
            if x_min <= x <= x_max and y_min <= y <= y_max:
                kps_[region_boundry_index].append(kp1[i])
                des_[region_boundry_index].append(des1[i])

    count = 0
    for kp in kps_:
       #print(len(kp))
        output_imaage = cv2.drawKeypoints(img1, kp,0,(0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("regional_kp_" + str(count) + ".jpg",output_imaage)
        count += 1
    return [kps_, des_]
def cutSmallestRegions(regions, num_to_cut=1):
    smallest = None
    for i in range(0, len(regions[0])):
        if smallest == None:
            smallest = [i, len(regions[0][i])]
        else:
            if len(regions[0][i]) < smallest[1]:
                smallest = [i, len(regions[0][i])]
    regions[0].pop(smallest[0])
    regions[1].pop(smallest[0])
    #print("region:" + str(len(regions[0])))
    return regions

def getHMatrixRegions(frames):
    H = None
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    # temporary for testing, set frames to first 2 frames in the input array
    # this should be changed later to fully support long arrays of frames
    img1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    mid_x = int(img1.shape[1] / 2)
    mid_y = int(img1.shape[0] / 2)


    #Below are keypoints for 4 segments
    #regions_boundries_1 = [((0,mid_x),(0,mid_y)),((mid_x,img1.shape[1]),(0,mid_y)),((0,mid_x),(mid_y,img1.shape[0])),((mid_x, img1.shape[1]),(mid_y,img1.shape[0]))]
    #regions_boundries_2 = [((0,mid_x),(0,mid_y)),((mid_x,img2.shape[1]),(0,mid_y)),((0,mid_x),(mid_y,img2.shape[0])),((mid_x, img2.shape[1]),(mid_y,img2.shape[0]))]
    boundary5 = (int(mid_x/2),int(mid_x + mid_x/2),int(mid_y/2),int(mid_y + mid_y/2))
    regions_boundries_1 = [(0, mid_x, 0, mid_y), (mid_x, img1.shape[1],0, mid_y),
                           (0, mid_x, mid_y, img1.shape[0]), (mid_x, img1.shape[1], mid_y, img1.shape[0]),boundary5]
    regions_boundries_2 = [(0,mid_x,0,mid_y),(mid_x,img2.shape[1],0,mid_y),(0,mid_x,mid_y,img2.shape[0]),(mid_x, img2.shape[1],mid_y,img2.shape[0]),boundary5]

    #Try slices instead
    #point_0 = 0
    #point_1 = img1.shape[1]/5
    #step = (img1.shape[1]/5)/2
    #regions_boundries_1 = []
    #regions_boundries_2 = []
    #while(point_1 <= img1.shape[0]):
    #    regions_boundries_1.append((0,img1.shape[1],point_0,point_1))
    #    #temp_step = step
    #    point_0 += step
    #    point_1 += step
    #if(point_1 < img1.shape[0]):
    #    regions_boundries_1.append((0,img1.shape[1],point_0,img1.shape[0]))

    regions_1 = partitionKeypoints2(img1, kp1, des1,regions_boundries_1)
    regions_1 = cutSmallestRegions(regions_1)
    #print(len(regions_1[0]))

    regions_2 = partitionKeypoints2(img2, kp2, des2,regions_boundries_1)
    regions_2 = cutSmallestRegions(regions_2)

    best_region = []
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.BFMatcher()
    count_1 = 0
    count_2 =0

    #reg_des_1 =np.array(reg_des_1)
    #reg_des_2 =np.array(reg_des_2)

        #region_1[1] = np.array(region_1[1])
    region_index_1 = 0
    region_index_2 = 0
    best_1 = -1
    best_2 = -1
    for region_1 in regions_1[1]:
        region_index_2 = 0

        test1 = np.array(region_1)
        for region_2 in regions_2[1]:

            test2 = np.array(region_2)
            if(region_index_1 != region_index_2):
                #region_2[1] = np.array(region_2[1])

                #print(type(des1[0]))
                #print(type(region_1[1][1]))
                matches = flann.knnMatch(test1, test2, k=2)
                # store all the good matches as per Lowe's ratio test.
                good = []
                H = None
                for m, n in matches:
                    if m.distance < 0.6 * n.distance:
                        good.append(m)
                if len(good) > len(best_region):
                    best_region = good
                    best_1 = region_index_1
                    best_2 = region_index_2
            region_index_2+=1
        region_index_1 +=1
    good = best_region


    best_region_1 = [regions_1[0][best_1], regions_1[1][best_1]]

    best_region_2 = [regions_2[0][best_2], regions_2[1][best_2]]

    if len(good) > MIN_MATCH_COUNT:
        src_pts = []
        dst_pts = []
        avg_x_change = 0
        avg_y_change = 0
        for m in good:

            src_pt = best_region_1[0][m.queryIdx].pt
            dst_pt = best_region_2[0][m.trainIdx].pt
            #src_pts.append(src_pt)
            #dst_pts.append(dst_pt)
            avg_x_change += abs(src_pt[0] - dst_pt[0])
            avg_y_change += abs(src_pt[1] - dst_pt[1])
        avg_x_change = avg_x_change/len(good)
        avg_y_change = avg_y_change/len(good)

        deviation = 10
        filter_count = 0
        for m in good:
            src_pt = best_region_1[0][m.queryIdx].pt
            dst_pt = best_region_2[0][m.trainIdx].pt
            x_change = abs(src_pt[0] - dst_pt[0])
            y_change = abs(src_pt[1] - dst_pt[1])
            if avg_y_change + deviation >= y_change >= avg_y_change - deviation:
                if avg_x_change + deviation >= x_change >= avg_x_change - deviation:
                    src_pts.append(src_pt)
                    dst_pts.append(dst_pt)
                    pass
            else:
                pass
        src_pts = np.float32(src_pts).reshape(-1,1,2)
        dst_pts = np.float32(dst_pts).reshape(-1,1,2)

        print(filter_count)
        #src_pts = np.float32([best_region_1[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
       # dst_pts = np.float32([best_region_2[0][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        img3 = cv2.drawMatches(img1, best_region_1[0], img2, best_region_2[0], good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("Matching_keypoints.jpg", img3)
        # H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        matchesMask = mask.ravel().tolist()
        # h, w, c = img2.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        # dst are the corners of the left image on the right image.

        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # cv2.imshow("Test", img2)
        # cv2.waitKey(0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return None
    return H
def getHMatrix(frames):
    H = None
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    #temporary for testing, set frames to first 2 frames in the input array
    #this should be changed later to fully support long arrays of frames
    img1 = frames[1]
    img2 = frames[0]
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.BFMatcher()
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    H = None
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("Matching_keypoints.jpg", img3)
        #H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        matchesMask = mask.ravel().tolist()
        #h, w, c = img2.shape
        #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        #dst = cv2.perspectiveTransform(pts, M)
        #dst are the corners of the left image on the right image.

        #img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        #cv2.imshow("Test", img2)
        #cv2.waitKey(0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return None
    return H
def openCVStitchImplentation(frames):

    MIN_MATCH_COUNT = 10
    #img1 = cv2.imread('box.png', 0)  # queryImage
    #img2 = cv2.imread('box_in_scene.png', 0)  # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    #temporary for testing, set frames to first 2 frames in the input array
    #this should be changed later to fully support long arrays of frames
    img1 = frames[1]
    img2 = frames[0]
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.BFMatcher()
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    M = None
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts,cv2.RANSAC, 3.0)

        #H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        matchesMask = mask.ravel().tolist()
        #h, w, c = img2.shape
        #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        #dst = cv2.perspectiveTransform(pts, M)
        #dst are the corners of the left image on the right image.

        #img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        #cv2.imshow("Test", img2)
        #cv2.waitKey(0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    dst = cv2.warpPerspective(img1, M, (img2.shape[1] +img1.shape[1] , img1.shape[0]))
    #out = cv2.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]), flags=cv2.INTER_LINEAR)
    #cv2.imshow('gray1.jpg',dst)
    #cv2.waitKey(0)

    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    return dst
    #cv2.imwrite('gray2.jpg',dst)
    #cv2.imshow("gray2", dst)

    #cv2.waitKey(0)
def stitchImagesRandomized(alpha, img_1, img_2):
    result = False
    runs = 0
    dst = None
    out_csv = "run,alpha,img1_detect,img1_compute,img1_keypoints,img2_detect,img_2_compute,img2_keypoints,keypoint_shift,copy,match_time,matches_found,filter,stitch_time,total_time\n"
    while not result and (runs <= alpha/1):
        runs += 1
        #print(runs)
        # Stitching code taken from https://towardsdatascience.com/image-stitching-using-opencv-817779c86a83
        # Modified to randomization
        img_ = img_1
        img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        img = img_2
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.ORB_create()
        # find the keypoints and descriptors with SIFT

        #  image2 = img[0:-1,img.shape[1]-200:-1]
        cols = int(img1.shape[0] * alpha)
        rows = int(img1.shape[1] * alpha)

        startX = random.randint(0, img1.shape[0] - cols)
        startY = random.randint(0, img1.shape[1] - rows)
        # startY = int(img1.shape[1] / 2 - 1)
        #print(startY)
        img2Holder = img2
        img2 = img2[0:-1, startY:startY + cols]
        # cv2.imshow("test", img2)
        # cv2.waitKey(0)
        startTime = time.time()
        kp1 = sift.detect(img1)
        img1_detect = (time.time() - startTime)
        startTime = time.time()

        kp1, des1 = sift.compute(img1, kp1)
        img1_compute = (time.time() - startTime)

        startTime = time.time()
        kp2 = sift.detect(img2)
        img2_detect = (time.time() - startTime)
        startTime = time.time()

        for kp in kp2:
            kp.pt = (kp.pt[0] + startY, kp.pt[1] + startX)
        keypoint_shift = (time.time() - startTime)

        startTime = time.time()
        img2 = img2Holder
        copy_time = (time.time() - startTime)

        startTime = time.time()

        kp2, des2 = sift.compute(img2, kp2)
        imgs2_compute = (time.time() - startTime)
        # kp2, des2 = sift.detectAndCompute(img2,None)

        bf = cv2.BFMatcher()
        startTime = time.time()
        if len(kp1) < 4 or len(kp2) < 4:
            break
        matches = bf.knnMatch(des1, des2, k=2)
        print(len(matches))
        match_time = startTime = time.time()

        # Apply ratio test
        startTime = time.time()

        good = []
        for m in matches:
            if m[0].distance < 0.7 * m[1].distance:
                good.append(m)
        matches = np.asarray(good)

        temp = None
        H = None
        if len(good) >= 4:
            result = True
            src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
            H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            #temp = H

            #cv2.invert(H,H)
            #temp[0][2] = H[0][2]
            #temp[1][2] = H[1][2]

            print("Test")
            #H[0][2] = H[0][2] * -1
            #H[1][2] = H[1][2] * -1

        else:
            filter_time = (time.time() - startTime)

            # update with process info so far here
            #    out_csv = "run,alpha,img1_detect,img1_compute,img1_keypoints,img2_detect,img_2_compute,img2_keypoints,keypoint_shift,copy,match_time,matches_found,filter,stitch_time, total_time"
            out_csv += str(runs) + "," + str(alpha) + "," + str(img1_detect) + "," + str(img1_compute) + "," + str(
                len(kp1)) + "," + str(img2_detect) + "," + str(imgs2_compute) + "," + str(len(kp2)) + "," + str(
                keypoint_shift) + "," + str(copy_time) + "," + str(match_time) + "," + str(len(matches)) + "," + str(
                filter_time) + "\n"
            continue
            # raise AssertionError("Can't find enough keypoints.")
        filter_time = (time.time() - startTime)

        full_keypoints = cv2.drawKeypoints(img_, kp1, 0, (0, 0, 255),
                                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        cv2.imwrite('output_fullKeypoints.jpg', full_keypoints)

        small_keypoints = cv2.drawKeypoints(img, kp2, 0, (0, 0, 255),
                                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('output_smallKeypoints.jpg', small_keypoints)

        img3 = cv2.drawMatchesKnn(img, kp1, img_, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        startTime = (time.time())
        #[, dst[, flags[, borderMode[, borderValue]]]]
        dst = cv2.warpPerspective(img_, H, (img.shape[1]+1000, img.shape[0]+1000))
        #cv2.imshow("test",img3)
        #cv2.waitKey(0)
        #cv2.imshow("subset",dst[0:img.shape[0], 0:img.shape[1]])
        cv2.imshow("img_",img_)
        cv2.imshow("img",img)
        cv2.imshow("dst",dst)
        dst[100:img.shape[0]+ 100, 100:img.shape[1] + 100] = img
        cv2.imshow("img_",img_)
        cv2.imshow("img",img)

        cv2.waitKey(0)
        stitch_time = (time.time() - startTime)
        cv2.imwrite('output_randomized.jpg', dst)
        cv2.imwrite("Matching_keypoints.jpg", img3)
        # out_csv = "run,alpha,img1_detect,img1_compute,img1_keypoints,img2_detect,img_2_compute,img2_keypoints,keypoint_shift,copy,match_time,matches_found,filter,stitch_time, total_time"
        #sum_time = img1_detect + img1_compute + img2_detect + imgs2_compute + keypoint_shift + copy_time + match_time + filter_time + stitch_time
        #out_csv += str(runs) + "," + str(alpha) + "," + str(img1_detect) + "," + str(img1_compute) + "," + str(
        #    len(kp1)) + "," + str(img2_detect) + "," + str(imgs2_compute) + "," + str(len(kp2)) + "," + str(
        #    keypoint_shift) + "," + str(copy_time) + "," + str(match_time) + "," + str(len(matches)) + "," + str(
        #    filter_time) + "," + str(stitch_time) + "," + str(sum_time) + "\n"
    #with open('output.csv', 'w') as outfile:
    #    outfile.write(out_csv)
    return (0,dst)


def stitchVideos(videos, fps):
    frame_skip = 800
    #frame_skip_cams = [2,0,0]

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    stitcher_time = ""
    fixed_H_time = ""
    hmatrixtime = ""
    #out_0 = cv2.VideoWriter("output_stitcher.mp4", fourcc, 10.0, (1280, 720))
    out_1 = cv2.VideoWriter("output_regional_h.mp4", fourcc, 10.0, (1280, 720))

    H = None
    stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
    caps = []
    for video in videos:
        caps.append((cv2.VideoCapture(video)))
    #cap1 = cv2.VideoCapture(video1)
    #cap2 = cv2.VideoCapture(video2)
    #cap3 = cv2.VideoCapture(video3)
    #for i in range(len(frame_skip_cams)):
    #    for j in range(0,frame_skip_cams[i]):
    #        caps[j].read()
    count = 0
    cams_up = True
    while cams_up:
        frames = []
        for cap in caps:
            holder = cap.read()
            #if frame collection fails
            if not holder[0]:
                cams_up = False
                break
            frames.append(holder[1])

        #ret2, frame2 = cap2.read()
        #ret3, frame3 = cap3.read()

        if count <=frame_skip:
            count += 1
            print(count)
            continue

        #count +=1
        if H is None:
            startTime = time.time()
            #H = getHMatrix(frames)
            H = getHMatrixRegions(frames)
            print("=====================\nTime: "+str(time.time() - startTime))
            init_stitch = fixHMatrixStitch(frames, H)
            cv2.imwrite("Init_stitch_sub.jpg", init_stitch)
            joined2 = np.concatenate(frames, axis=1)
            cv2.imwrite("joined_cameras.jpg", joined2)
        count +=1
        #startTime = time.time()

        #try:
        #    joined = stitcher.stitch([frames[0], frames[1]])
        #except cv2.error:
        #    print("boop")
        #stitcher_time += str(time.time() - startTime) + "\n"

        #result = stitchImagesRandomized(0.5,frames[1], frames[2])
        #cv2.imshow("half",result[1])
        startTime = time.time()
        #openCVStitchImplentation(frames)
        fixed_h_result = fixHMatrixStitch(frames, H)
        fixed_H_time += str(time.time() - startTime) + "\n"
        #if(joined[1] is not None):
        #    joined_out = cv2.resize(joined[1], (1280,720))
        #    out_0.write(joined_out)
        startTime = time.time()
        #H = getHMatrixRegions(frames)
        #hmatrixtime += str(time.time() - startTime) + "\n"
        #gen_image = fixHMatrixStitch(frames,H)
        #cv2.imwrite("frames_new\\image_" + str(count)+".jpg", fixed_h_result)
        fixed_h_result = cv2.resize(fixed_h_result, (1280,720))
        out_1.write(fixed_h_result)
        #cv2.namedWindow("joined_cameras", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("joined2", cv2.WINDOW_NORMAL)
       # cv2.imwrite("Cameraas_side_by_side.jpg", joined2)
        #cv2.imwrite("local_class_frame_1_2.jpg", joined)
        #cv2.imshow("joined_cameras", joined)
        #print(joined)
        #cv2.imshow("joined",joined[1])

        #cv2.imshow("joined2",result[1])
        #cv2.waitKey(0)
        if count > frame_skip+1000:
            break
    with open('stitcher_times.csv', 'w') as outfile:
        outfile.write(stitcher_time)
    with open('fixed_h_times.csv', 'w') as outfile:
        outfile.write(fixed_H_time)
    #with open('h_matrix_gen_time.csv', 'w') as outfile:
    #    outfile.write(hmatrixtime)

def main():
    stitchVideos([r".\\take1_videos\\output2.avi",r".\\take1_videos\\output1.avi",  r".\\take1_videos\\output3.avi"], 15)
    #stitchVideos([r".\\take_2_videos\\output_0.mp4",r".\\take_2_videos\\output_1.mp4",  r".\\take_2_videos\\output_2.mp4"], 15)

if __name__ == "__main__":
    main()
