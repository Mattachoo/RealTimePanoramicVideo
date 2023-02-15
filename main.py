import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import ImageValidator
import cv2
import numpy as np
from matplotlib import pyplot as plt


# PriorityQueue adapted from https://www.geeksforgeeks.org/priority-queue-in-python/
class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data):
        max_val = -1
        if len(self.queue) < 1:
            self.queue.append(data)
            return
        for i in range(len(self.queue)):
            if self.queue[i][0] > data[0]:
                self.queue.insert(i, data)
                return
        self.queue.append(data)

    # for popping an element based on Priority
    def pop(self):
        return self.queue.pop(0)

    def deleted(self, index):
        return self.queue.pop(index)

    def peek(self):
        if self.isEmpty():
            return [[0]]
        try:
            return self.queue[0]
        except:
            return [[0]]


def findInitialHomography():
    pass


def mergeImges():
    pass


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop top
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop top
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def get_overlap_corners(H, img1, img2, shapes):
    h, w, c = img2.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(pts, H)
    x_min = max(corners[0][0][0], corners[1][0][0])
    x_max = min(corners[2][0][0], corners[3][0][0])
    y_min = max(corners[0][0][1], corners[3][0][1])
    y_max = min(corners[1][0][1], corners[2][0][1])

    return (int(x_min), int(y_min), int(x_max), int(y_max))


def generate_masks(H, img1, img2, shapes):
    h, w, c = shapes[0]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(pts, H)
    # print(corners)
    # compare [0][0] and [1][0], take greater value
    # compare [2][0] and [3][0], take smaller value
    # compare [0][1] and [3][1], take greater value
    # compare [1][1] amd [2][1], take smaller value

    x_min = max(corners[0][0][0], corners[1][0][0])
    x_max = min(corners[2][0][0], corners[3][0][0])
    y_min = max(corners[0][0][1], corners[3][0][1])
    y_max = min(corners[1][0][1], corners[2][0][1])

    dst = cv2.warpPerspective(img1, H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
    img2 = cv2.copyMakeBorder(img2, 0, 0, 0, 500, cv2.BORDER_CONSTANT)
    overlap = cv2.cvtColor(cv2.multiply(img2, dst), cv2.COLOR_BGR2GRAY)

    size = shapes[0][0], shapes[0][1] + 500
    size_2 = shapes[0][0], shapes[0][1] + 500, 3

    # cv2.imshow("dst", overlap)
    # cv2.waitKey(0)
    overlap_coords = cv2.findNonZero(overlap)
    test = cv2.UMat.get(overlap)
    overlap_coords = cv2.UMat.get(overlap_coords)
    sub_arr_mask = np.zeros(size, dtype=np.uint8)
    inv_arr_mask = np.ones(size, dtype=np.uint8) * 255
    # inv_arr_mask = np.ones(size, dtype=np.uint8)
    sub_arr = np.zeros(size_2, dtype=np.uint8)
    inv_arr = np.ones(size_2, dtype=np.uint8) * 255
    # inv_arr = np.ones(size_2, dtype=np.uint8)
    inv_arr_2 = np.ones(size_2, dtype=np.uint8)
    # cv2.imshow("dst",dst)
    for coord in overlap_coords:
        y = coord[0][1]
        x = coord[0][0]
        inv_arr_2[y][x] = [0, 0, 0]

        if x_min <= x <= x_max and y_min <= y <= y_max:
            sub_arr[y][x] = [255, 255, 255]
            sub_arr_mask[y][x] = 255
            inv_arr[y][x] = [0, 0, 0]
            inv_arr_mask[y][x] = 0
    # sub_arr = cv2.bitwise_not(inv_arr)
    # cv2.imshow("sub_arr",sub_arr)
    # cv2.imshow("inv_arr",inv_arr_mask)

    # cv2.waitKey(0)
    return sub_arr, inv_arr, inv_arr_mask, inv_arr_2


def getOverlapRegionCornerBased(img1, img2, shapes, H, overlap_mask, inv_arr, inv_arr_mask):
    h, w, c = shapes[0]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(pts, H)
    # print(corners)
    # compare [0][0] and [1][0], take greater value
    # compare [2][0] and [3][0], take smaller value
    # compare [0][1] and [3][1], take greater value
    # compare [1][1] amd [2][1], take smaller value

    x_min = max(corners[0][0][0], corners[1][0][0])
    x_max = min(corners[2][0][0], corners[3][0][0])
    y_min = max(corners[0][0][1], corners[3][0][1])
    y_max = max(corners[1][0][1], corners[2][0][1])
    # rect = cv2.rectangle((x_min,y_min),())

    # rect = (x_min,y_min, x_max, y_max)
    rect = (x_min, y_min, x_max, y_max)

    # img_test = img1(rect)
    # dst_sz = cv2.detail.resultRoi(corners=corners, sizes=shapes[0:2 ])
    blend_strength = 1
    #    blend_width = np.sqrt()
    blender = cv2.detail_MultiBandBlender(try_gpu=1)

    # blender.setNumBands()
    overlap_mask = cv2.cvtColor(overlap_mask, cv2.COLOR_BGR2GRAY)
    inv_arr = cv2.cvtColor(inv_arr, cv2.COLOR_BGR2GRAY)
    s = (shapes[0][1] + 500, shapes[0][0])
    # overlap_mask_inv = np.ones((s[1],s[0]))*255-overlap_mask
    # overlap_mask_inv = cv2.cvtColor(overlap_mask_inv, cv2.COLOR_BGR2GRAY)
    im_r_mask = np.concatenate((np.zeros(s), np.ones(s) * 255), axis=1).astype('uint8')
    im_l_mask = np.concatenate((np.ones(s) * 255, np.zeros(s)), axis=1).astype('uint8')
    dst = cv2.warpPerspective(img1, H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
    # blender.feed(dst, overlap_mask, (corners[0][0][0],corners[0][0][1]))
    # blender.feed(img2, overlap_mask, (corners[0][0][0],corners[0][0][1]))
    res = (np.zeros((shapes[0][1] + 500, 3))).astype('uint8')
    res_mask = (np.ones((shapes[0][0], shapes[0][1] + 500, 3)) * 255).astype('uint8')
    res_mask_temp = cv2.cvtColor(res_mask, cv2.COLOR_BGR2GRAY)
    blender.prepare((0, 0, 1780, 720))
    startTime = time.time()
    blender.feed(dst, overlap_mask, (rect[0], rect[1]))
    blender.feed(img2, inv_arr, (rect[0], rect[1]))
    test = None
    test_mask = None
    out, out_mask = blender.blend(test, test_mask)
    # print(time.time() - startTime)
    # plt.imshow(cv2.cvtColor(out.astype('uint8'), cv2.COLOR_BGR2RGB))
    # plt.show()
    return np.array(out)


# img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
def getOverlapRegionMultBased(frames, H, shapes, sub_arr, inv_arr, inv_arr_2):
    img1 = frames[1]
    img2 = frames[0]
    # add_one = np.ones((shapes[0][0], shapes[0][1], 3), dtype=np.uint8)

    # img2 = cv2.add(img2, add_one)
    # img1 = cv2.add(img1, add_one)

    # startTime = time.time()
    img2_border = cv2.copyMakeBorder(img2, 0, 0, 0, 500, cv2.BORDER_CONSTANT)
    alpha = 0.1
    beta = 1 - alpha
    gamma = 0

    dst = cv2.warpPerspective(img1, H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
    # startTime = time.time()
    # getOverlapRegionCornerBased(dst, img1, img2,shapes,H, sub_arr, inv_arr,inv_arr_mask)
    # print(time.time() - startTime)
    # combined = cv2.bitwise_or(dst, img2_border, mask=inv_arr_mask)
    combined = cv2.add(cv2.multiply(dst, inv_arr_2), cv2.multiply(img2_border, inv_arr))
    img_overlap = cv2.multiply(img2_border, sub_arr)

    # get overallping area in dst
    dst_overlap = cv2.multiply(dst, sub_arr)
    dst = cv2.addWeighted(dst_overlap, alpha, img_overlap, beta, gamma)

    # cv2.imshow("combined",combined)

    # cv2.imshow("dst",dst)
    # cv2.waitKey(0)
    dst = cv2.add(dst, combined)

    # print(time.time() - startTime)
    return dst


def getOverlapRegionCudaBased(frames, H, shapes, sub_arr, inv_arr, inv_arr_2):
    # img1 = cv2.cuda_GpuMat()
    # img2 = cv2.cuda_GpuMat()
    # img1.upload(frames[1])
    # img2.upload(frames[0])
    img1 = frames[1]
    img2 = frames[0]
    # add_one = np.ones((shapes[0][0], shapes[0][1], 3), dtype=np.uint8)

    # img2 = cv2.add(img2, add_one)
    # img1 = cv2.add(img1, add_one)

    # startTime = time.time()
    img2_border = cv2.cuda.copyMakeBorder(img2, 0, 0, 0, 500, cv2.BORDER_CONSTANT)
    alpha = 0.5
    beta = 1 - alpha
    gamma = 0

    dst = cv2.warpPerspective(img1, H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
    # startTime = time.time()
    # getOverlapRegionCornerBased(dst, img1, img2,shapes,H, sub_arr, inv_arr,inv_arr_mask)
    # print(time.time() - startTime)
    # combined = cv2.bitwise_or(dst, img2_border, mask=inv_arr_mask)
    combined = cv2.cuda.add(cv2.cuda.multiply(dst, inv_arr_2), cv2.cuda.multiply(img2_border, inv_arr))
    img_overlap = cv2.cuda.multiply(img2_border, sub_arr)

    # get overallping area in dst
    dst_overlap = cv2.cuda.multiply(dst, sub_arr)
    dst = cv2.cuda.addWeighted(dst_overlap, alpha, img_overlap, beta, gamma)

    # cv2.imshow("combined",combined)

    # cv2.imshow("dst",dst)
    # cv2.waitKey(0)
    dst = cv2.cuda.add(dst, combined)

    # print(time.time() - startTime)
    return dst


def seamEstimation(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    seam_x = -1
    best_column = None
    column_width = int(img2.shape[1] / 10)
    curr_x = 0
    sums = []
    while curr_x < img2.shape[1]:
        result = 0
        result = (img1_gray[:, curr_x:curr_x + column_width] - img2_gray[:, curr_x:curr_x + column_width]).sum()
        # result2 = img1[:,curr_x:curr_x+column_width].sum() -img2[:,curr_x:curr_x+column_width].sum()
        # result = np.mean()
        sums.append(result)
        # print(result)

        curr_x += column_width
    # print(sums)
    # cv2.imshow("result",result)
    # cv2.waitKey(0)


def blendWeightedCustom(img1, img2, alpha, beta, gamma):
    result = img1
    for row in range(0, img2.shape[0] - 1):
        for col in range(0, img2.shape[1] - 1):
            pixel = 0
            if img1[row][col][0] == [0] and img1[row][col][1] == 0:
                pixel = img2[row][col]
            elif img2[row][col][0] == 0 and img2[row][col][1] == 0:
                pixel = img1[row][col]
            else:
                pixel = img1[row][col] * alpha + img2[row][col] * beta + gamma
            result[row][col] = pixel
    cv2.imshow("result", result)
    cv2.waitKey(0)
    return result


def calc_sloped_coord(coor_0, coor_1, x_value):
    slope = (coor_0[1] - coor_1[1]) / (coor_0[0] - coor_1[0])
    y_value = slope * (x_value - coor_1[0]) + coor_1[1]
    return x_value, y_value


def get_seams(img1, img2, seam_size, factors, bounds):
    pos_bot, pos_top = bounds
    factors_left, factors_right = factors
    if pos_top[1] <= 0:
        pos_top = (pos_top[0], 0)
    startTime = time.time()

    seam_1 = img1[int(pos_top[1]):int(pos_bot[1]), int(img2.shape[1]) - seam_size:int(img2.shape[1])]
    seam_2 = img2[int(pos_top[1]):int(pos_bot[1]), int(img2.shape[1]) - seam_size:int(img2.shape[1])]

    seam_join = edgeWeightedBlending(seam_2, seam_1, factors_left, factors_right)

    # print(time.time() - startTime)
    result = img1
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    result[int(pos_top[1]):int(pos_bot[1]), int(img2.shape[1]) - seam_size:int(img2.shape[1])] = seam_join
    # cv2.imshow("dst",img1)
    # cv2.waitKey(0)
    return result


def get_seams_parallel(img1, img2, seam_size, factors, bounds, priorityList):
    pos_bot, pos_top = bounds
    factors_left, factors_right = factors

    startTime = time.time()

    seam_1 = img1[int(pos_top[1]):int(pos_bot[1]), int(img2.shape[1]) - seam_size:int(img2.shape[1])]
    seam_2 = img2[int(pos_top[1]):int(pos_bot[1]), int(img2.shape[1]) - seam_size:int(img2.shape[1])]
    seam_join = edgeWeightedBlending(seam_2, seam_1, factors_left, factors_right)

    print(time.time() - startTime)
    result = img1
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    result[int(pos_top[1]):int(pos_bot[1]), int(img2.shape[1]) - seam_size:int(img2.shape[1])] = seam_join
    # cv2.imshow("dst",img1)
    # cv2.waitKey(0)
    return result


def cpuStitchParallel(frames, H, shapes, seam_size, factors, bounds, priority_list):
    img1 = frames[1]
    img2 = frames[0]
    # seam_pts = get_overlap_corners(H, img1, img2, shapes)
    dst = cv2.warpPerspective(img1, H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
    result = get_seams(dst, img2, seam_size, factors, bounds)

    return result


def fixHMatrixStitch(frames, H, shapes, seam_size, factors, bounds):
    # startTime = time.time()
    dst = cpuStitch(frames, H, shapes, seam_size, factors, bounds)
    # dst = getOverlapRegionMultBased(frames, H, shapes, sub_arr, inv_arr, inv_arr_2)
    # dst = getOverlapRegionCornerBased(frames[1], frames[0], shapes, H, sub_arr, inv_arr, inv_arr_mask)
    # dst = getOverlapRegionCudaBased(frames, H, shapes, sub_arr, inv_arr, inv_arr_2)
    # print(time.time() - startTime)
    return dst


def fixHMatrixStitchGPU(frames, GPU_frames, H):
    # test = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
    # img1 =cv2.cuda_GpuMat
    # img1.upload(test)
    # img2 =cv2.cuda_GpuMat
    # img2.upload(frames[0])

    img1 = GPU_frames[1]
    img2 = GPU_frames[0]
    test = img2.size()
    # print(type(test))
    startStitchtime = time.time()
    dst = cv2.cuda.warpPerspective(img1, H, (img2.size()[0] + img1.size()[0], img1.size()[1]))
    shift = 0
    # temp_img_1 = img1.download()
    # temp_img_2 = img2.download()
    dst_temp = dst.download()
    # download_time = time.time() -start_time
    # print(str(download_time))
    dst_temp[0:frames[0].shape[0], 0 + shift:frames[0].shape[1] + shift] = frames[0]
    # print(time.time() - startStitchtime - download_time)
    return dst


def partitionKeypoints(img1, kp1, des1):
    regions = [[], [], [], []]
    out_kp = []
    out_des = []
    kps_ = [[], [], [], []]
    des_ = [[], [], [], []]
    for i in range(0, len(kp1)):
        x = kp1[i].pt[0]
        y = kp1[i].pt[1]
        mid_x = int(img1.shape[1] / 2)
        mid_y = int(img1.shape[0] / 2)
        # Top left
        if x < mid_x:
            # Top Left
            if y < mid_y:
                kps_[0].append(kp1[i])
                des_[0].append(des1[i])

                # regions[0].append([kp1[i], des1[i]])
            # Bottom Left
            else:
                kps_[2].append(kp1[i])
                des_[2].append(des1[i])
                # regions[2].append([kp1[i], des1[i]])
        else:
            # Top Right
            if y < mid_y:
                kps_[1].append(kp1[i])
                des_[1].append(des1[i])
                # regions[1].append([kp1[i], des1[i]])
            # Bottom Right
            else:
                kps_[3].append(kp1[i])
                des_[3].append(des1[i])
                # regions[3].append([kp1[i], des1[i]])
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
        # print(len(kp))
        output_imaage = cv2.drawKeypoints(img1, kp, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("regional_kp_" + str(count) + ".jpg", output_imaage)
        count += 1
    return [kps_, des_]


def partitionKeypoints2(img1, kp1, des1, boundries):
    # regions = [[]] * len(boundries)
    out_kp = []
    out_des = []

    kps_ = []
    des_ = []

    for boundry in boundries:
        kps_.append([])
        des_.append([])
    for i in range(0, len(kp1)):
        x = kp1[i].pt[0]
        y = kp1[i].pt[1]
        # Below section needs to be updated to use given regions
        for region_boundry_index in range(0, len(boundries)):

            x_min = boundries[region_boundry_index][0]
            x_max = boundries[region_boundry_index][1]
            y_min = boundries[region_boundry_index][2]
            y_max = boundries[region_boundry_index][3]
            # mid_x = int(img1.shape[1]/2)
            # mid_y = int(img1.shape[0]/2)
            # Top left
            if x_min <= x <= x_max and y_min <= y <= y_max:
                kps_[region_boundry_index].append(kp1[i])
                des_[region_boundry_index].append(des1[i])

    count = 0
    for kp in kps_:
        # print(len(kp))
        output_imaage = cv2.drawKeypoints(img1, kp, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("regional_kp_" + str(count) + ".jpg", output_imaage)
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
    # print("region:" + str(len(regions[0])))
    return regions


def verticalSlices(shapes, regions_boundries_1):
    point_0 = 0
    point_1 = shapes[0][0] / 5
    step = (shapes[0][0] / 5) / 2

    while (point_1 <= shapes[0][0]):
        regions_boundries_1.append((point_0, point_1, 0, shapes[0][0]))
        # temp_step = step
        point_0 += step
        point_1 += step
    if (point_1 < shapes[0][0]):
        regions_boundries_1.append((point_0, shapes[0][0], 0, shapes[0][0]))


def getHMatrix(frames):
    H = None
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    # temporary for testing, set frames to first 2 frames in the input array
    # this should be changed later to fully support long arrays of frames
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
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("Matching_keypoints.jpg", img3)
        matchesMask = mask.ravel().tolist()

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return None
    return H


def stitchImagesRandomized(alpha, img_1, img_2):
    result = False
    runs = 0
    dst = None
    out_csv = "run,alpha,img1_detect,img1_compute,img1_keypoints,img2_detect,img_2_compute,img2_keypoints,keypoint_shift,copy,match_time,matches_found,filter,stitch_time,total_time\n"
    while not result and (runs <= alpha / 1):
        runs += 1
        # print(runs)
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
        # print(startY)
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
        # print(len(matches))
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
            # temp = H

            # cv2.invert(H,H)
            # temp[0][2] = H[0][2]
            # temp[1][2] = H[1][2]

            print("Test")
            # H[0][2] = H[0][2] * -1
            # H[1][2] = H[1][2] * -1

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
        # [, dst[, flags[, borderMode[, borderValue]]]]
        dst = cv2.warpPerspective(img_, H, (img.shape[1] + 1000, img.shape[0] + 1000))
        # cv2.imshow("test",img3)
        # cv2.waitKey(0)
        # cv2.imshow("subset",dst[0:img.shape[0], 0:img.shape[1]])
        cv2.imshow("img_", img_)
        cv2.imshow("img", img)
        cv2.imshow("dst", dst)
        dst[100:img.shape[0] + 100, 100:img.shape[1] + 100] = img
        cv2.imshow("img_", img_)
        cv2.imshow("img", img)

        cv2.waitKey(0)
        stitch_time = (time.time() - startTime)
        cv2.imwrite('output_randomized.jpg', dst)
        cv2.imwrite("Matching_keypoints.jpg", img3)
        # out_csv = "run,alpha,img1_detect,img1_compute,img1_keypoints,img2_detect,img_2_compute,img2_keypoints,keypoint_shift,copy,match_time,matches_found,filter,stitch_time, total_time"
        # sum_time = img1_detect + img1_compute + img2_detect + imgs2_compute + keypoint_shift + copy_time + match_time + filter_time + stitch_time
        # out_csv += str(runs) + "," + str(alpha) + "," + str(img1_detect) + "," + str(img1_compute) + "," + str(
        #    len(kp1)) + "," + str(img2_detect) + "," + str(imgs2_compute) + "," + str(len(kp2)) + "," + str(
        #    keypoint_shift) + "," + str(copy_time) + "," + str(match_time) + "," + str(len(matches)) + "," + str(
        #    filter_time) + "," + str(stitch_time) + "," + str(sum_time) + "\n"
    # with open('output.csv', 'w') as outfile:
    #    outfile.write(out_csv)
    return (0, dst)


def edgeWeightedBlendingColumnsOnly(left_img, right_img):
    try:
        if left_img.shape != left_img.shape:
            raise
        # We need to weigh the blending by how close the pixel is to each edge
        # If close to the left side, we weigh more heavily in favor of the left image, and the same logic for the right
        # For now, we can just treat it as if there are only left and right sides, top and bottom can be done later
        result = left_img
        edge_blend_window = 200
        columns = left_img.shape[1]
        rows = left_img.shape[0]
        for i in range(0, columns):
            # factor = ((columns -i)/columns + (rows - j)/rows)/2
            factor = (columns - i) / columns
            result[:, i] = left_img[:, i] * factor + right_img[:, i] * (1 - factor)
        # cv2.imshow("left", left_img)
        return result
    except():
        print("Matrices are not the same shape")


def computeBlendingMatrix(img_shape):
    row_blend_span = 10
    # result = left_img
    edge_blend_window = 200
    columns = img_shape[1]
    rows = img_shape[0]
    factors_left = np.zeros(img_shape)
    factors_right = np.zeros(img_shape)
    for i in range(0, columns):
        for j in range(0, rows):
            factor = 1
            # if j <= rows/2:
            #    factor = ((columns -i)/columns + (rows - j)/rows)/2
            # factor = ((columns -i)/columns + (rows - j)/rows)/2
            # elif j >= rows/2:
            #    factor =((columns -i)/columns +  (1- (rows - j)/rows))/2

            # else:
            factor = (columns - i) / columns
            factors_left[j][i] = (factor, factor, factor)
            factors_right[j][i] = (1 - factor, 1 - factor, 1 - factor)
    return factors_left, factors_right


def computeBlendingMatrixBidirectional(img_shape):
    row_blend_span = 10

    # result = left_img
    edge_blend_window = 200
    columns = img_shape[1]
    rows = img_shape[0]
    factors_left = np.zeros(img_shape)
    factors_right = np.zeros(img_shape)
    for i in range(0, columns):
        for j in range(0, rows):
            factor = 1
            if j <= rows / 2:
                factor = ((columns - i) / columns + (rows - j) / rows) / 2
            elif j >= rows / 2:
                factor = ((columns - i) / columns + (1 - (rows - j) / rows)) / 2

            else:
                factor = (columns - i) / columns
            factors_left[j][i] = (factor, factor, factor)
            factors_right[j][i] = (1 - factor, 1 - factor, 1 - factor)

    return factors_left, factors_right


def edgeWeightedBlending(left_img, right_img, factors_left, factors_right):
    try:
        if left_img.shape != left_img.shape:
            raise
        # We need to weigh the blending by how close the pixel is to each edge
        # If close to the left side, we weigh more heavily in favor of the left image, and the same logic for the right
        # For now, we can just treat it as if there are only left and right sides, top and bottom can be done later
        result = left_img * factors_left + right_img * factors_right

        return result
    except():
        print("Matrices are not the same shape")


def preprocessing(frames_og, shapes, seam_size):
    H = getHMatrixRegions(frames_og, shapes)
    # H[0] performs X shift
    # H[1][0] peforms y hift
    # H[2] does something
    # H[2][1] = H[2][1] +0.5
    bounds = calc_sloped_coords(frames_og[0], H)
    pos_bot, pos_top = bounds
    dst = cv2.warpPerspective(frames_og[1], H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
    seam_1 = dst[int(pos_top[1]):int(pos_bot[1]), int(frames_og[0].shape[1]) - seam_size:int(frames_og[0].shape[1])]
    factors = computeBlendingMatrix(seam_1.shape)
    init_stitch = cpuStitch(frames_og, H, shapes, seam_size, factors, bounds)
    cv2.imwrite("Init_stitch_sub.jpg", init_stitch)
    # joined2 = cv2.hconcat(frames)
    # cv2.imwrite("joined_cameras.jpg", joined2)
    cv2.imwrite("init_stitch.jpg", init_stitch)
    return H, bounds, factors


def perspectiveWarp(frames, shapes, H, seam_size, factors, bounds, priorityList):
    dst = cv2.warpPerspective(frames[1], H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
    # result = get_seams(dst, frames[0], seam_size, factors, bounds)
    return dst
    # priorityList[2].insert(0,(get_seams_parallel,(dst, frames[0], seam_size, factors, bounds,priorityList)))


def performWarpBlendStitchSaveFrames(frames_og, shapes, H, seam_size, factors, bounds, count, buffer):
    startTime = time.time()
    # dst = cv2.warpPerspective(frames_og[1], H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
    result = cpuStitch(frames_og, H, shapes, seam_size, factors, bounds)
    performWarpBlendStitchTime = str(time.time() - startTime)
    # print(performWarpBlendStitchTime)
    # buffer.insert((count, cv2.resize(result, (1280, 720))))
    return cv2.resize(result, (1280, 720))


def count_frames(videos):
    frame_skip = 1000
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    stitcher_time = ""
    fixed_H_time = ""
    hmatrixtime = ""
    seam_size = 50
    factors = []
    bounds = []
    out_1 = cv2.VideoWriter("output_regional_h.mp4", fourcc, 10.0, (1280, 720))
    H = None
    caps = []
    frame_count = []
    for video in videos:
        caps.append((cv2.VideoCapture(video)))
        frame_count.append(0)
    count = 0
    cams_up = True
    index = 0
    for cap in caps:
        while True:
            ret, frame = cap.read()
            if ret:
                frame_count[index] += 1
            else:
                print()
                index += 1
                break
    print(frame_count)


def generate_training_data(videos):
    print(os.cpu_count())

    frame_skip = 100
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    stitcher_time = ""
    fixed_H_time = ""
    hmatrixtime = ""
    seam_size = 50
    factors = []
    bounds = []
    out_1 = cv2.VideoWriter("output_regional_h.mp4", fourcc, 10.0, (1280, 720))
    H = None
    caps = []
    for video in videos:
        caps.append((cv2.VideoCapture(video)))
    count = 0
    cams_up = True
    buffer = PriorityQueue()
    priorityList = []
    next_tag = 0
    startTime2 = time.time()

    startTime2 = time.time()
    frame_num = 0
    while cams_up:
        frames = []
        shapes = []
        frames_og = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                cams_up = False
                # print("end of video stream")
                break
            shapes.append(frame.shape)
            frames_og.append(frame)
        if count <= frame_skip:
            count += 1
            next_tag = count
            continue

        if H is None:
            # place this at the end of the highest priority list, since it should always be run asap when needed
            # priorityList[0].append((preprocessing,[frames_og,shapes, seam_size]))
            # H, bounds, factors = preprocessing(frames_og,shapes, seam_size)

            # H, bounds, factors=preprocessing(frames_og,shapes, seam_size)
            H, bounds, factors = preprocessing(frames_og, shapes, seam_size)
            print(H)
            # buffer_runner = executor.submit(checkBuffer, *[buffer, next_tag, out_1])

        # if SVM return false
        # spawn a preprocesing thread

        startTime = time.time()
        print(type(H))
        H_temp = np.copy(H)

        # for x_shift in range(-10,10,3):
        #     #print(H)
        #     H_temp[1][0] =   H[1][0] + x_shift/100
        #     for y_shift in range(-10,10,3):
        #         H_temp[0][0] =  H[0][0]  + y_shift/100
        #        if y_shift != 0 or x_shift !=0:
        result = performWarpBlendStitchSaveFrames(frames_og, shapes, H_temp, seam_size, factors, bounds, count, buffer)
        cv2.imwrite("videos_og\\frame_" + str(frame_num) + ".jpg", result)
        # H_temp = H

        # fixed_h_result = performWarpBlendStitch(frames_og, shapes, H, seam_size,factors,bounds,count,buffer)
        # out_1.write(fixed_h_result)
        frame_num += 1
        count += 1
        # if count > frame_skip + 10000:
        #    break

    # with open('h_matrix_gen_time.csv', 'w') as outfile:
    #    outfile.write(hmatrixtime)


#    H, bounds, factors = preprocessing(frames_og, shapes, seam_size)

class VideoStitcher:
    def __init__(self):
        self.seam_size = 20
        self.H = None
        self.logreg = None
        self.bounds = None
        self.factors = None
        self.dopreprocessing = True
        self.best_score = 0
        self.best_diff = -1
        self.current_frames = None
        self.corners = None
        self.left_side_mask = None
        self.right_side_mask = None
        self.threshold = 0
        self.img_scores_dict = None

    def validateStitch(self, buffer, images, shapes, seam_size, threshold):

        image = cv2.resize(cv2.cvtColor(buffer.peek()[1], cv2.COLOR_BGR2GRAY), (600, 400)).flatten()

        score = self.logreg.predict([image])[0]
        # print(self.logreg.predict([image]))
        # print("score:",score)
        if self.best_score == 0:
            self.best_score = score
        if score < threshold:
            print("Score:", score, "falls below threshold", threshold)
            self.dopreprocessing = True
            self.best_score = score
        elif score < self.best_score:
            print("Score:", score, "is better than best_score:", self.best_score)
            self.dopreprocessing = True
            self.best_score = score

    def gen_score(self, frames, H, seam_size):
        pos_bot, pos_top = self.bounds
        dst = cv2.warpPerspective(frames[1], H, (frames[0].shape[1] + 500, frames[0].shape[0]),
                                  borderMode=cv2.BORDER_CONSTANT)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        # img = frames[0]
        seam_1 = dst[int(pos_top[1]):int(pos_bot[1]), int(frames[0].shape[1]) - seam_size:int(frames[0].shape[1])]
        seam_2 = img[int(pos_top[1]):int(pos_bot[1]), int(img.shape[1]) - seam_size:int(img.shape[1])]

        # seam_1, seam_2 = adaptive_thresholding([seam_1,seam_2],cv2.ADAPTIVE_THRESH_MEAN_C)
        return cv2.absdiff(seam_1, seam_2).sum() / (seam_2.shape[0] + seam_2.shape[1])
        # print("score:",score)
        # print(score)

    def gen_score_superpixels(self, frames, H, seam_size):
        pos_bot, pos_top = self.bounds
        dst = cv2.warpPerspective(frames[1], H, (frames[0].shape[1] + 500, frames[0].shape[0]),
                                  borderMode=cv2.BORDER_CONSTANT)
        # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # img = frames[0]
        seam_1 = dst[int(pos_top[1]):int(pos_bot[1]), int(frames[0].shape[1]) - seam_size:int(frames[0].shape[1])]
        seam_2 = img[int(pos_top[1]):int(pos_bot[1]), int(img.shape[1]) - seam_size:int(img.shape[1])]

        # seam_1, seam_2 = adaptive_thresholding([seam_1,seam_2],cv2.ADAPTIVE_THRESH_MEAN_C)
        return cv2.absdiff(seam_1, seam_2).sum() / (seam_2.shape[0] + seam_2.shape[1])
        # print("score:",score)
        # print(score)

    def img_scores_sum(self, img_scores_dict):
        sum = 0
        for key in img_scores_dict.keys():
            sum += img_scores_dict[key]
        return sum

    def validateStitchDiffs(self, buffer, shapes, seam_size, threshold):
        timeout2 = 0
        # print("Hit")
        while timeout2 < 100:
            # print(type(self.current_frames))
            if self.current_frames is not None:
                timeout2 = 0

                # frames = buffer.peek()[2]
                frames = self.current_frames
                new_H = self.getHMatrixRegions(frames, shapes)
                # score = self.img_scores_sum(self.img_scores_dict)
                if new_H is None:
                    continue
                img1 = frames[1]
                img2 = frames[0]
                # print("score:",score)
                # print(score)

                # print("Best Diff:", self.best_diff, "VS New Diff:",score)
                dst = cv2.warpPerspective(img2, new_H, (img1.shape[1] + 500, img1.shape[0]),
                                          borderMode=cv2.BORDER_CONSTANT)
                if self.best_diff == -1:
                    score = self.img_scores_sum(self.img_scores_dict)
                    self.best_diff = score
                    continue

                score_new = math.inf
                if np.sum(new_H) > 0:
                    # score_new = self.gen_score(frames, new_H, seam_size)
                    h, w, c = img2.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                    corners = cv2.perspectiveTransform(pts, new_H)
                    x_min = max(self.corners[0][0][0], self.corners[1][0][0])
                    x_max = min(self.corners[2][0][0], self.corners[3][0][0])
                    y_min = max(self.corners[0][0][1], self.corners[3][0][1])
                    y_max = min(self.corners[1][0][1], self.corners[2][0][1])

                    seam_1 = img1[:, int(x_min):int(x_max - x_min)]
                    seam_2 = dst[:, int(x_min):int(x_max - x_min)]
                    img_scores_dict_new, superpixel_graph, superpixels = self.superpixel_cost_estimation(seam_1, seam_2)
                    score_new = self.img_scores_sum(img_scores_dict_new)

                    if score_new < score:
                        print("found better score:", score_new, "Better than:", score)
                        self.best_diff = score_new
                        self.H = new_H
                        self.preprocessing(frames, shapes, seam_size)
                        continue

                elif self.best_diff > score:
                    print("Diff Change: ", self.best_diff, "->", score)
                    self.best_diff = score
                    self.dopreprocessing = True
                time.sleep(1)
            else:
                # print("validation waiting")
                timeout2 += 1
                time.sleep(0.1)
        print("validation time out")

    def superpixel_stitch(self, img1, dst, seam_size, factors, bounds):

        x_min = max(self.corners[0][0][0], self.corners[1][0][0])
        x_max = min(self.corners[2][0][0], self.corners[3][0][0])
        y_min = max(self.corners[0][0][1], self.corners[3][0][1])
        y_max = min(self.corners[1][0][1], self.corners[2][0][1])
        # print(self.left_side_mask)
        seam_1 = img1[:, int(x_min):int(self.left_side_mask.shape[1] + x_min)]
        seam_2 = dst[:, int(x_min):int(self.left_side_mask.shape[1] + x_min)]
        # print(seam_1.shape)
        # print(self.left_side_mask.shape)
        #cv2.imshow("Test", seam_1 * self.left_side_mask)
        #cv2.waitKey(0)
        joined_img = (seam_1 * self.left_side_mask + seam_2 * self.right_side_mask)
        joined_img = joined_img.astype('uint8')
        #print(joined_img.dtype)
        #print(seam_1.dtype)
        # cv2.cvtColor(joined_img, cv2.COLOR_BGRA2RGB)
        # joined_img = cv2.bitwise_or(right_side, left_side)
        # cv2.imshow("seam_1", seam_1)
        # cv2.imshow("seam_2", seam_2)
        # cv2.imshow("Dst", dst)
        ##cv2.imshow('img1',img1)

        #cv2.imshow("Joined", joined_img)
        #cv2.waitKey(0)
        result = dst
        result[:, int(x_min):int(x_max - x_min)] = joined_img
        result[:, 0:(int(x_min))] = img1[:, 0:(int(x_min))]

        # result = get_seams(dst, img1, seam_size, factors, bounds)
        # seam_join = edgeWeightedBlending(seam_2, seam_1, self.factors_left, self.factors_right)

        # cv2.imshow("Result", result)
        # cv2.waitKey(0)

        return result

    def cpuStitch(self, frames, H, shapes, seam_size, factors, bounds):
        img1 = frames[1]
        img2 = frames[0]
        # seam_pts = get_overlap_corners(H, img1, img2, shapes)
        startTime = time.time()
        dst = cv2.warpPerspective(img1, H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
        warpTime = time.time() - startTime
        # result = get_seams(dst, img2, seam_size, factors, bounds)
        result = self.superpixel_stitch(img2, dst, seam_size, factors, bounds)
        # print(time.time() - startTime)
        return result

    def performWarpBlendStitch(self, frames_og, shapes, H, seam_size, factors, bounds, count, buffer):
        startTime = time.time()
        # dst = cv2.warpPerspective(frames_og[1], H, (shapes[0][1] + 500, shapes[0][0]), borderMode=cv2.BORDER_CONSTANT)
        result = self.cpuStitch(frames_og, H, shapes, seam_size, factors, bounds)
        performWarpBlendStitchTime = str(time.time() - startTime)
        # print(performWarpBlendStitchTime)
        if buffer is not None:
            buffer.insert((count, cv2.resize(result, (1280, 720))))
            self.current_frames = frames_og
        else:
            cv2.imshow("blend_stitch_result", cv2.resize(result, (1280, 720)))
            cv2.waitKey(0)
        # if self.validation_counter % self.validation_interval == 0:
        #    self.validateStitch(result,frames_og,shapes,seam_size,0.9)

    # return cv2.resize(result, (1280, 720))
    def calc_sloped_coords(self, img2, H):
        h, w, c = img2.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        self.corners = cv2.perspectiveTransform(pts, H)
        # note order is [x,y]
        bottom_right = self.corners[2][0]
        bottom_left = self.corners[1][0]
        top_left = self.corners[0][0]
        top_right = self.corners[3][0]

        pos_top = calc_sloped_coord(top_left, top_right, w)
        pos_bot = calc_sloped_coord(bottom_left, bottom_right, w)
        return pos_bot, pos_top

    def preprocessing(self, frames_og, shapes, seam_size):
        print("running preprocessing")
        startTime = time.time()
        self.H = self.getHMatrixRegions(frames_og, shapes)
        # print("H Mat done")
        if np.sum(self.H) == 0:
            return
        # print(time.time() - startTime)
        # H[0] performs X shift
        # H[1][0] peforms y hift
        # H[2] does something
        # H[2][1] = H[2][1] +0.5

        self.bounds = self.calc_sloped_coords(frames_og[0], self.H)
        # print(self.bounds)
        dst = cv2.warpPerspective(frames_og[1], self.H, (shapes[0][1] + 500, shapes[0][0]),
                                  borderMode=cv2.BORDER_CONSTANT)
        pos_bot, pos_top = self.bounds
        if pos_top[1] <= 0:
            pos_top = (pos_top[0], 0)
        seam_1 = dst[int(pos_top[1]):int(pos_bot[1]), int(frames_og[0].shape[1]) - seam_size:int(frames_og[0].shape[1])]
        seam_2 = frames_og[0][int(pos_top[1]):int(pos_bot[1]),
                 int(frames_og[0].shape[1]) - seam_size:int(frames_og[0].shape[1])]
        # print(seam_1.shape)
        # print(seam_2.shape)
        self.factors = computeBlendingMatrix(seam_1.shape)
        # print("RUnning create_seam_masks")
        self.create_seam_masks(frames_og[0], frames_og[1])
        print("Done with create_seam_masks")
        init_stitch = self.cpuStitch(frames_og, self.H, shapes, seam_size, self.factors, self.bounds)
        print("Cpu stitch done")
        cv2.imwrite("Init_stitch_sub.jpg", init_stitch)
        cv2.imwrite("init_stitch.jpg", init_stitch)

    def redo_preprocessing(self, frames_og, shapes, seam_size):
        print("running redo_preprocessing")
        startTime = time.time()
        if np.sum(self.H) == 0:
            return
        print(time.time() - startTime)
        # H[0] performs X shift
        # H[1][0] peforms y hift
        # H[2] does something
        # H[2][1] = H[2][1] +0.5

        self.bounds = self.calc_sloped_coords(frames_og[0], self.H)
        dst = cv2.warpPerspective(frames_og[1], self.H, (shapes[0][1] + 500, shapes[0][0]),
                                  borderMode=cv2.BORDER_CONSTANT)
        pos_bot, pos_top = self.bounds

        seam_1 = dst[int(pos_top[1]):int(pos_bot[1]), int(frames_og[0].shape[1]) - seam_size:int(frames_og[0].shape[1])]
        seam_2 = frames_og[0][int(pos_top[1]):int(pos_bot[1]),
                 int(frames_og[0].shape[1]) - seam_size:int(frames_og[0].shape[1])]
        # print("Diff:",self.compare_images(seam_1,seam_2))
        self.factors = computeBlendingMatrix(seam_1.shape)
        init_stitch = self.cpuStitch(frames_og, self.H, shapes, seam_size, self.factors, self.bounds)
        cv2.imwrite("Init_stitch_sub.jpg", init_stitch)
        # joined2 = cv2.hconcat(frames)
        # cv2.imwrite("joined_cameras.jpg", joined2)
        cv2.imwrite("init_stitch.jpg", init_stitch)
        # return bounds, factors

    def checkBuffer(self, buffer, next_tag, out_1):
        timeout = 0
        timeout2 = 0
        time_cap = 1000
        while timeout < 100:
            # print("timeout")
            if not buffer.isEmpty():
                timeout = 0
                timeout2 += 1
                if buffer.peek()[0] == next_tag:
                    timeout2 = 0

                    out_1.write(buffer.pop()[1])
                    # print(buffer)
                    if (next_tag % 100 == 0):
                        print(next_tag)
                        # pass
                    next_tag += 1
                if timeout2 > time_cap:
                    print("time_cap")
                    timeout2 = 0
                    new_frame = buffer.pop()
                    next_tag = new_frame[0] + 1
                    out_1.write(new_frame[1])

            else:
                # yprint("buffer waiting")
                time.sleep(0.01)
                timeout += 1
        self.current_frames = None

    # intented to be run over the seams
    def compare_images(self, img1, img2):
        img1_copy = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_copy = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img_1_border = [0, 0, 0, 0]
        img_2_border = [0, 0, 0, 0]
        if img1.shape[0] > img2.shape[0]:
            pass
        elif img1.shape[1] > img2.shape[1]:
            pass
        if img1.shape[0] > img2.shape[0]:
            pass
        elif img2.shape[1] < img2.shape[1]:
            pass
        diff = cv2.absdiff(img1_copy, img2_copy)
        print(diff)
        return diff.sum()

    def stitchVideos(self, videos, fps):
        print(os.cpu_count())
        frame_skip = 1000
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        stitcher_time = ""
        fixed_H_time = ""
        hmatrixtime = ""
        seam_size = 50
        # factors = []
        # bounds = []
        out_1 = cv2.VideoWriter("output_regional_h.mp4", fourcc, 10.0, (1280, 720))
        # H = None
        caps = []
        validation_counter = 0
        for video in videos:
            caps.append((cv2.VideoCapture(video)))
        count = 0
        cams_up = True
        buffer = PriorityQueue()
        priorityList = []
        next_tag = 0
        startTime2 = time.time()
        self.validation_counter = 0
        self.validation_interval = 100
        initial = True
        with ThreadPoolExecutor(max_workers=12) as executor:
            startTime2 = time.time()
            self.runnning = True
            while cams_up:
                frames = []
                shapes = []
                frames_og = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        cams_up = False
                        # print("end of video stream")
                        break
                    shapes.append(frame.shape)
                    frames_og.append(frame)
                if count <= frame_skip:
                    count += 1

                    next_tag = count
                    # print(count)
                    continue

                if self.H is None or self.dopreprocessing:
                    # place this at the end of the highest priority list, since it should always be run asap when needed
                    # priorityList[0].append((preprocessing,[frames_og,shapes, seam_size]))
                    # H, bounds, factors = preprocessing(frames_og,shapes, seam_size)

                    # H, bounds, factors=preprocessing(frames_og,shapes, seam_size)
                    self.preprocessing(frames_og, shapes, seam_size)
                    if np.sum(self.H) > 0:
                        self.dopreprocessing = False
                    # print(self.H)
                    else:
                        continue
                if initial:
                    executor.submit(self.validateStitchDiffs, *[buffer, shapes, seam_size, 0.9])
                    buffer_runner = executor.submit(self.checkBuffer, *[buffer, next_tag, out_1])
                    initial = False
                # if SVM return false
                # spawn a preprocesing thread

                startTime = time.time()
                executor.submit(self.performWarpBlendStitch,
                                *[frames_og, shapes, self.H, seam_size, self.factors, self.bounds, count, buffer])
                self.validation_counter += 1
                if self.validation_counter % 100 == 0:
                    self.validation_counter = 0

                    # validateStitch(self, buffer, images, shapes, seam_size, threshold):

                    # executor.submit(self.validateStitch, *[buffer, frames_og, shapes, seam_size, 0.9])

                # self.validateStitch(buffer, frames_og, shapes, seam_size, 0.9)
                count += 1
            buffer_runner.result()

            print("Buffer done")
            out_1.release()
            executor.shutdown()
        print(time.time() - startTime2)
        with open('stitcher_times.csv', 'w') as outfile:
            outfile.write(stitcher_time)
        with open('fixed_h_times.csv', 'w') as outfile:
            outfile.write(fixed_H_time)
        # with open('h_matrix_gen_time.csv', 'w') as outfile:
        #    outfile.write(hmatrixtime)

    def getHMatrixRegions(self, frames, shapes):
        # print("Start getHMatrixRegions")
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

        regions_boundries_1 = []

        point_0 = 0
        point_1 = img1.shape[0] / 4
        step = (img1.shape[0] / 3) / 2
        # horitzontal slices
        while (point_1 <= img1.shape[0]):
            regions_boundries_1.append((0, img1.shape[1], point_0, point_1))
            # temp_step = step
            point_0 += step
            point_1 += step
        if (point_1 < img1.shape[0]):
            regions_boundries_1.append((0, img1.shape[1], point_0, img1.shape[0]))

        regions_1 = partitionKeypoints2(img1, kp1, des1, regions_boundries_1)
        regions_1 = cutSmallestRegions(regions_1)
        # print(len(regions_1[0]))

        regions_2 = partitionKeypoints2(img2, kp2, des2, regions_boundries_1)
        regions_2 = cutSmallestRegions(regions_2)

        best_region = []
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.BFMatcher()
        count_1 = 0
        count_2 = 0

        region_index_1 = 0
        region_index_2 = 0
        best_1 = -1
        best_2 = -1
        i = 0
        j = 0
        for region_1 in regions_1[1]:
            region_index_2 = 0

            j = 0
            test1 = np.array(region_1)
            for region_2 in regions_2[1]:
                j += 1
                test2 = np.array(region_2)
                if (region_index_1 != region_index_2):
                    # region_2[1] = np.array(region_2[1])

                    # print(type(des1[0]))
                    # print(type(region_1[1][1]))
                    # print(str(i) + " : " + str(j))
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
                region_index_2 += 1
            i += 1
            region_index_1 += 1
        good = best_region

        best_region_1 = [regions_1[0][best_1], regions_1[1][best_1]]

        best_region_2 = [regions_2[0][best_2], regions_2[1][best_2]]
        distance = []
        if len(good) > MIN_MATCH_COUNT:
            src_pts = []
            dst_pts = []
            avg_x_change = 0
            avg_y_change = 0
            for m in good:
                distance.append(m.distance)
                src_pt = best_region_1[0][m.queryIdx].pt
                dst_pt = best_region_2[0][m.trainIdx].pt

                avg_x_change += abs(src_pt[0] - dst_pt[0])
                avg_y_change += abs(src_pt[1] - dst_pt[1])
            distance = np.array(distance)

            avg_x_change = avg_x_change / len(good)
            avg_y_change = avg_y_change / len(good)

            deviation = 10
            filter_count = 0
            # print("Good: ", len(good))
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

            src_pts = np.float32(src_pts).reshape(-1, 1, 2)
            dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
            if (len(src_pts) >= MIN_MATCH_COUNT):

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                img3 = cv2.drawMatches(img1, best_region_1[0], img2, best_region_2[0], good, None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite("Matching_keypoints.jpg", img3)
                matchesMask = mask.ravel().tolist()

            else:
                print("Not enough points")
                return None
                H = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None
            return None
        # print("End getHMatrixRegions")

        return H

    def gen_label_dict(self, img_labels):

        img_dict = {}
        # img_dict = {coords, bounds}
        print("labels:", len(img_labels[0]))
        curr_superpixel = -1
        start_pos = 0
        # img_dict[y][x]
        for y in range(0, len(img_labels)):
            for x in range(0, len(img_labels[0])):
                value = img_labels[y][x]

                if img_labels[y][x] not in img_dict.keys():
                    # print(value)
                    img_dict[value] = [[(y, x)], []]
                    if curr_superpixel == -1:
                        curr_superpixel = value
                        start_pos = x
                    else:
                        img_dict[curr_superpixel][1].append((start_pos, x - 1, y))
                        if x == 0:
                            print("sp dict:", img_dict[curr_superpixel][0])
                            print("superpixel:", curr_superpixel, ",", (start_pos, x - 1), "Row: ", y)

                        curr_superpixel = value
                        start_pos = x
                else:

                    img_dict[value][0].append((y, x))
                    if curr_superpixel != value:

                        if curr_superpixel == -1:
                            curr_superpixel = value
                            start_pos = x
                        else:
                            img_dict[curr_superpixel][1].append((start_pos, x - 1, y))
                            # print("sp dict:",  img_dict[curr_superpixel][0])
                            # print("superpixel2:", curr_superpixel, ",",img_dict[curr_superpixel][1][-1][:2], "Row: ", img_dict[curr_superpixel][1][-1][2])
                            curr_superpixel = value
                            start_pos = x

            img_dict[value][1].append((start_pos, x, y))
            curr_superpixel = -1
            # print("superpixel:", value, ",", (start_pos, i))

            start_pos = 0
        return img_dict
        # print(len(slic_result.getLabels()))

    def get_superpixel_scores(self, img1, label_dict, img2):
        img_scores_dict = {}

        for key in label_dict.keys():
            sum = 0
            for coord in label_dict[key][0]:
                # get absdiff of rpgs values, then add them together
                sum += np.sum(cv2.absdiff(img1[coord[0]][coord[1]], img2[coord[0]][coord[1]]))
                # img_scores_dict[key] = sum
                # sum += item
            img_scores_dict[key] = (sum / len(label_dict[key][0])) / 765
        # print(img_scores_dict)
        return img_scores_dict

    def superpixel_cost_estimation(self, img1, img2):
        # take pixel values of super pixel cell, sum them up, subtract by value from corresponding image
        # perform same operation for edges, using histogram of gradient descents
        # take sum of both. Result is the cost of that given cell
        # print("img2 shape", img2.shape)

        img1_superpixels = cv2.ximgproc.createSuperpixelSLIC(img1)
        img1_superpixels.iterate(50)

        img1_num_of_superpixels = img1_superpixels.getNumberOfSuperpixels()
        img1_superpixel_labels = img1_superpixels.getLabels()
        img1_superpixel_mask = img1_superpixels.getLabelContourMask()

        img2_superpixels = cv2.ximgproc.createSuperpixelSLIC(img2)
        img2_superpixels.iterate(50)

        img2_num_of_superpixels = img2_superpixels.getNumberOfSuperpixels()
        img2_superpixel_labels = img2_superpixels.getLabels()
        img3_superpixel_mask = img2_superpixels.getLabelContourMask()

        # to make processing easier, we can create a dictionary for each image, which has a label as its key, and a list of coordiantes are items
        img1_dict = self.gen_label_dict(img1_superpixel_labels)
        # img2_dict = self.gen_label_dict(img2_superpixel_labels)
        # Scores closer to 1 are more similar

        print(img1_dict[1][1])
        img_scores_dict = self.get_superpixel_scores(img1, img1_dict, img2)

        # img2[0:img1.shape[0], 0:img1.shape[1]] = img1
        # print("Gray shape:", cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
        # diffIntensity = (cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
        # print("Shape:",diffIntensity.shape)
        # for key in img1_dict.keys():
        #    count = 0
        #    for coord in img1_dict[key][1]:
        #        #print("Diff shape", diffIntensity.shape)
        #        diffIntensity[:,:coord[1]]
        #        cv2.imshow("Diff Intensity2", diffIntensity[:,:coord[1]])
        #        cv2.waitKey(0)
        #        count +=1
        # diffIntensity[coord[0]][coord[1]] = img_scores_dict[key] * 255
        # print(diffIntensity)
        # diffIntensity = cv2.cvtColor(diffIntensity, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Diff Intensity", img1)
        ##cv2.imshow("Diff Intensity2", img2)
        # cv2.waitKey(0)
        superpixel_graph = self.build_superpixel_graph2(img1_superpixel_labels, img_scores_dict)
        # print("SP Grah:", superpixel_graph)
        return img_scores_dict, superpixel_graph, img1_superpixels, img1_dict
        # print(len(img1_dict[0]))

    def run(self):
        # self.logreg = ImageValidator.get_model(["videos_og", "videos_og_1"], "videos_shifted", 1000)

        #self.stitchVideos([r".\\take_1_trimmed\\output_1.mp4", r".\\take_1_trimmed\\output_0.mp4",
        #                   r".\\take_1_trimmed\\output_2.mp4"], 15)

        self.stitchVideos([r".\\rain_recording\\output_0.mp4", r".\\rain_recording\\output_1.mp4",r".\\rain_recording\\output_2.mp4"], 15)

    # below is test code for single set of frames
    def create_seam_masks(self, img1, img2):
        # print("Gen Masks Start")

        startTime = time.time()
        dst = cv2.warpPerspective(img2, self.H, (img1.shape[1] + 500, img1.shape[0]), borderMode=cv2.BORDER_CONSTANT)

        x_min = max(self.corners[0][0][0], self.corners[1][0][0])
        x_max = min(self.corners[2][0][0], self.corners[3][0][0])
        y_min = max(self.corners[0][0][1], self.corners[3][0][1])
        y_max = min(self.corners[1][0][1], self.corners[2][0][1])

        seam_1 = img1[:, int(x_min):int(x_max - x_min)]
        seam_2 = dst[:, int(x_min):int(x_max - x_min)]
        self.img_scores_dict, superpixel_graph, seam1_supers, img1_dict = self.superpixel_cost_estimation(seam_1,
                                                                                                          seam_2)
        lowest_cost_path = self.find_lowest_cost_path(self.img_scores_dict, superpixel_graph, seam1_supers)
        lowest_cost_path_img = np.zeros(seam_1.shape)
        lowest_cost_path_inv = np.zeros(seam_1.shape)
        # = 0
        startTime2 = time.time()
        row_dict = {}
        row = 0
        rightmost = 0
        center = 0.5
        steps = int(self.seam_size / 2)
        print("steps", steps)
        test = computeBlendingMatrix((1, self.seam_size, 3))

        for superpixel in lowest_cost_path[1]:
            # print(superpixel, ":")
            for coord in img1_dict[superpixel][1]:
                # print(coord[2])
                # print("Superpixel:", superpixel, "Row:", coord)
                if coord[2] not in row_dict:
                    row_dict[coord[2]] = coord[1]
                else:
                    if row_dict[coord[2]] < coord[1]:
                        row_dict[coord[2]] = coord[1]
        for row in row_dict.keys():
            lowest_cost_path_img[row, :row_dict[row]] = 1
            lowest_cost_path_inv[row, row_dict[row]:] = 1
            print(test.shape)
            lowest_cost_path_inv[row, row_dict[row] - int(self.seam_size / 2): row_dict[row]+int(self.seam_size / 2)] = test[1][0:lowest_cost_path_inv[row, row_dict[row] - int(self.seam_size / 2): row_dict[row]+int(self.seam_size / 2)].shape[1]]
            lowest_cost_path_img[row, row_dict[row] - int(self.seam_size / 2): row_dict[row]+int(self.seam_size / 2)] = test[0][0:lowest_cost_path_img[row, row_dict[row] - int(self.seam_size / 2): row_dict[row]+int(self.seam_size / 2)].shape[1]]

            # lowest_cost_path_img[row, row_dict[row]] = 0.5
            # lowest_cost_path_inv[row, row_dict[row]] = 0.5
            # for step in range(0,int(steps/2)):
            #    value = (center + step/steps)
            #    lowest_cost_path_img[row,row_dict[row]-step] = value
            #    lowest_cost_path_inv[row,row_dict[row]+step] = value

        print("Gen masks time:", time.time() - startTime2)

        self.left_side_mask = lowest_cost_path_img
        self.right_side_mask = lowest_cost_path_inv
        print("create_seam_masks processing time:", time.time() - startTime)

    def shift_and_seam(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        shapes = [img1.shape, img2.shape]
        frames_og = [img1, img2]
        seam_size = 50
        self.seam_size = seam_size
        self.preprocessing(frames_og, shapes, seam_size)
        startTime = time.time()
        slic_result = cv2.ximgproc.createSuperpixelSLIC(img1)
        slic_result.iterate(50)
        slic_mask = slic_result.getLabelContourMask()
        masked_img1 = cv2.bitwise_and(img1, img1, mask=cv2.bitwise_not(slic_mask))
        dst = cv2.warpPerspective(img2, self.H, (img1.shape[1] + 500, img1.shape[0]), borderMode=cv2.BORDER_CONSTANT)

        h, w, c = img2.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        corners = cv2.perspectiveTransform(pts, self.H)

        x_min = max(corners[0][0][0], corners[1][0][0])
        x_max = min(corners[2][0][0], corners[3][0][0])
        y_min = max(corners[0][0][1], corners[3][0][1])
        y_max = min(corners[1][0][1], corners[2][0][1])

        pos_bot, pos_top = self.bounds
        factors_left, factors_right = self.factors

        startTime = time.time()
        seam_size = 500

        seam_1 = img1[int(y_min):int(y_max), int(x_min):int(x_max - x_min)]
        seam_2 = dst[int(y_min):int(y_max), int(x_min):int(x_max - x_min)]
        img_scores_dict, superpixel_graph, superpixels = self.superpixel_cost_estimation(seam_1, seam_2)
        lowest_cost_path = self.find_lowest_cost_path(img_scores_dict, superpixel_graph, superpixels)
        lowest_cost_path_img = (seam_1) * 0
        seam1_supers = cv2.ximgproc.createSuperpixelSLIC(seam_1)
        seam1_supers.iterate(50)
        # print(lowest_cost_path[1])
        # print(len(seam1_supers.getLabels())," : ",len(seam1_supers.getLabels()[0]))
        startTime2 = time.time()
        for y in range(0, len(seam1_supers.getLabels())):
            # print(y)
            for x in range(0, len(seam1_supers.getLabels()[0])):
                # print(seam1_supers.getLabels()[x][y])
                # print(x)
                if seam1_supers.getLabels()[y][x] in lowest_cost_path[1]:
                    lowest_cost_path_img[y][x] = (255, 255, 255)
                    # print("breaking")
                    break
                lowest_cost_path_img[y][x] = (255, 255, 255)
        print("Gen masks time:", time.time() - startTime2)
        # print("Best Path")
        # thresh, lowest_cost_path_img = cv2.threshold(lowest_cost_path_img, 175, 255, cv2.THRESH_BINARY)
        # print(lowest_cost_path_img.shape, "VS", seam_1.shape)

        # print(cv2.bitwise_and(lowest_cost_path_img,seam_1))
        # print((lowest_cost_path_img/255)* seam_1)
        lowest_cost_path_inv = cv2.bitwise_not(lowest_cost_path_img)
        print("Graph Estimation Time:", time.time() - startTime)

        startTime = time.time()

        left_side = cv2.bitwise_and(seam_1, lowest_cost_path_img)
        right_side = cv2.bitwise_and(seam_2, lowest_cost_path_inv)

        joined_img = cv2.bitwise_or(right_side, left_side)
        print("Time elasped:", time.time() - startTime)
        # cv2.imshow("Left Side", seam_1)
        # cv2.imshow("Right Side", seam_2 )
        cv2.imshow("Best Path", cv2.bitwise_or(left_side, right_side))

        cv2.waitKey(0)
        # for item in (slic_result.getLabels()[1]):
        #    print(item)

    #        cv2.imshow("slice_result",slic_result.getLabels())
    # cv2.imshow("Masked image", masked_img1)
    # cv2.waitKey(0)

    def find_lowest_cost_path(self, img_scores_dict, superpixel_graph, superpixels):
        # We want to start from superpixel which lay on the top of the overlapping area
        sp_dict = {}
        for superpixel in set(superpixels.getLabels().flatten()):
            sp_dict[superpixel] = [math.inf, []]
        top_superpixels = set(superpixels.getLabels()[0])
        # print("Top Superpixels:", top_superpixels)
        for superpixel in top_superpixels:
            sp_dict[superpixel][0] = 0
            # print(sp_dict[superpixel][1])
            sp_dict[superpixel][1] = [superpixel]
            self.recursive_find(img_scores_dict, sp_dict, superpixel, superpixel_graph)
        # print(sp_dict)
        bottom_superpixels = set(superpixels.getLabels()[-1])
        lowest_cost_path = [math.inf, set()]

        for bottom_superpixel in bottom_superpixels:
            if sp_dict[bottom_superpixel][0] < lowest_cost_path[0]:
                lowest_cost_path = sp_dict[bottom_superpixel]
        # print(lowest_cost_path)
        return lowest_cost_path

    def recursive_find(self, img_scores, sp_dict, source, superpixel_graph):
        vertices = superpixel_graph[source]
        for vertex in vertices:
            cost = img_scores[vertex] + sp_dict[source][0]
            if cost < sp_dict[vertex][0]:
                sp_dict[vertex][0] = cost
                sp_dict[vertex][1] = np.append(sp_dict[source][1], vertex)
                self.recursive_find(img_scores, sp_dict, vertex, superpixel_graph)

    def build_superpixel_graph(self, superpixels, cost_graph):
        superpixel_graph = {}
        # We can say there is a path between superpixels if they are adjacent to eaachother
        for x in range(0, len(superpixels)):
            print(len(superpixels[0]))
            for y in range(0, len(superpixels[x])):
                if superpixels[x][y] not in superpixel_graph.keys():
                    superpixel_graph[superpixels[x][y]] = set()
                # if new super pixel to the right, add to set
                if x + 1 < len(superpixels):
                    if superpixels[x][y] != superpixels[x + 1][y]:
                        superpixel_graph[superpixels[x][y]].add(superpixels[x + 1][y])
                    if y + 1 < len(superpixels[x]):
                        if superpixels[x][y] != superpixels[x + 1][y + 1]:
                            superpixel_graph[superpixels[x][y]].add(superpixels[x + 1][y + 1])
                if 0 <= x - 1:
                    if superpixels[x][y] != superpixels[x - 1][y]:
                        superpixel_graph[superpixels[x][y]].add(superpixels[x - 1][y])
                    if superpixels[x][y] != superpixels[x - 1][y + 1]:
                        superpixel_graph[superpixels[x][y]].add(superpixels[x - 1][y + 1])
        return superpixel_graph

    def build_superpixel_graph2(self, superpixels, cost_graph):
        superpixel_graph = {}
        # We can say there is a path between superpixels if they are adjacent to eaachother
        # print("num of supers:", superpixels[-1][-1])
        # Loop through Row
        curr_id = -1
        for y in range(0, len(superpixels[0])):
            # loop through column
            curr_id = -1
            for x in range(0, len(superpixels)):
                # if y+1 < len(superpixels[0]):
                if superpixels[x][y] != curr_id and curr_id != -1:
                    if curr_id not in superpixel_graph:
                        superpixel_graph[curr_id] = set(curr_id)

                    else:
                        superpixel_graph[curr_id].add(superpixels[x][y])
                curr_id = superpixels[x][y]
                if curr_id not in superpixel_graph:
                    superpixel_graph[curr_id] = set()

        # for key in superpixel_graph.keys():
        #    print(key, ":", superpixel_graph[key])
        # print("graph size: ", len(superpixel_graph))

        return superpixel_graph


def adaptive_thresholding(imgs, type):
    threshold_imgs = []
    # print("test")
    for image in imgs:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if (cv2.THRESH_OTSU):
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold_imgs.append(th)

        else:
            # print("test2")
            img = cv2.medianBlur(img, 5)
            th = cv2.adaptiveThreshold(img, 255, type, cv2.THRESH_BINARY, 11, 2)
            threshold_imgs.append(th)
    # print("Test3")
    return threshold_imgs
    # th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


def main():
    # generate_training_data(
    #    [r".\\take_1_trimmed\\output_1.mp4", r".\\take_1_trimmed\\output_0.mp4", r".\\take_1_trimmed\\output_2.mp4"])
    # validator = ImageValidator()
    # logreg = ImageValidator.get_model(["videos_og", "videos_og_1"], "videos_shifted", 1000)
    # stitchVideos([r".\\take_1_trimmed\\output_1.mp4", r".\\take_1_trimmed\\output_0.mp4", r".\\take_1_trimmed\\output_2.mp4"], 15)
    stitcher = VideoStitcher()
    # frame_1100.jpg produces a pretty good stitch
    frame = 1100

    base_dir = r"C:\Users\mattp\PycharmProjects\pythonProject\video_to_frame_take1_trimmed"
    # stitcher.shift_and_seam(base_dir + r"\video_1\\frame_" + str(frame) + ".jpg",
    #                        base_dir + r"\video_0\\frame_" + str(frame - 1) + ".jpg")
    stitcher.run()


# count_frames(
#     [r".\\take_1_trimmed\\output_1.mp4", r".\\take_1_trimmed\\output_0.mp4", r".\\take_1_trimmed\\output_2.mp4"])


if __name__ == "__main__":
    main()
