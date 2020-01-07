"""
Author: Zhibo Fan
Implementation for SIFT/SURF match and segmentation overlapping
to verify finger vein images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import roi
import itertools
import sklearn.mixture as mix
import skimage.transform as skit
import os

# Change this DATA_PATH to your custom path format!
DATA_PATH = 'class1/ep{}/{}/{}-{}-{}-1.bmp'

feature = cv2.xfeatures2d.SIFT_create()
# feature = cv2.xfeatures2d.SURF_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

def sift_match(roi1, roi2, thresh=0.6, vis=False):
    """
    Using CLAHE to augment images for better sift features.
    A good sift keypoint match is defined as followed:
    First find top two best matches with the FlannMatcher, 
    if the best match is significantly better than the second best,
    then the best match should be a good match.
    This could prevent threshold variance for different images.
    """
    img1 = roi.clahe(roi1, gray=True)
    img2 = roi.clahe(roi2, gray=True)
    kp1, des1 = feature.detectAndCompute(img1, None)
    kp2, des2 = feature.detectAndCompute(img2, None)
    matches = flann.knnMatch(des1, des2, k=2) 
    good_matches = [[m[0]] for m in matches if m[0].distance < thresh * m[1].distance]
    if vis:
        _roi1 = cv2.drawKeypoints(img1, kp1, np.array([]), color=(255, 0, 0))
        _roi2 = cv2.drawKeypoints(img2, kp2, np.array([]), color=(255, 0, 0))
        cv2.imwrite('sift_points1.jpg', _roi1)
        cv2.imwrite('sift_points2.jpg', _roi2)
        match_res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('sift_match.jpg', match_res)
    return len(good_matches) / max(len(kp1), len(kp2))
    
def sift_process(path1, path2):
    roi1 = roi.process(path1)
    roi2 = roi.process(path2)
    return sift_match(roi1, roi2)
    
def segm_match(roi1, roi2, vis=True):
    """
    Given the observation that the augmented grayscale image of finger vein
    may contain three different colored areas representing for background, 
    shadow around the veins and vein areas, a GMM is applied for adaptive
    multi-threshold segmentation. Thus in this scenario matched pixel rate
    (MPR) should be identical to mask iou.
    """
    gray1, gray2 = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in [roi1, roi2]]
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    gray1 = cv2.filter2D(gray1, -1, kernel)
    gray2 = cv2.filter2D(gray2, -1, kernel)
    gmm = mix.GaussianMixture(n_components=3)
    gmm.fit(gray1.reshape(-1, 1))
    lable1 = gmm.predict(gray1.reshape(-1, 1)).reshape(gray1.shape)
    minlable = min([(i, x) for i, x in enumerate(gmm.means_[:, 0])], key=lambda x: x[1])[0]
    lable1 = (lable1 == minlable).astype(np.uint8)
    gmm.fit(gray2.reshape(-1, 1))
    lable2 = gmm.predict(gray2.reshape(-1, 1)).reshape(gray2.shape)
    minlable = min([(i, x) for i, x in enumerate(gmm.means_[:, 0])], key=lambda x: x[1])[0]
    lable2 = (lable2 == minlable).astype(np.uint8)
    shape = (64, 64)
    lable1 = cv2.resize(lable1, shape)
    lable2 = cv2.resize(lable2, shape)
    area = np.sum(lable1) + np.sum(lable2)
    inter = np.sum(lable1 * lable2)
    iou = inter / (area + inter)
    if vis:
        cv2.imwrite('temp.jpg', cv2.resize(lable2, lable1.shape).astype(int) * 255)
        cv2.imwrite('segm_match.jpg', lable1 * 255)
    return iou
    
def segm_process(path1, path2):
    roi1 = roi.process(path1)
    roi2 = roi.process(path2)
    return segm_match(roi1, roi2)
    
def calc_stat(stu=None, proc=sift_process, pos_only=False, neg_only=False, limit=None):
    """
    This function collects all the MPR statistics for certain image pairs. They are collected into
    positive and negative groups and stored as npy files. To visualize, see plot function below.
    :param: STU is the index of a student or None for computing all given students.
    :param: PROC is the function parameter, SIFT_PROCESS or SEGM_PROCESS
    :param: if POS_ONLY, only positive matching statistics are calculated. Same for NEG_ONLY.
    :param: LIMIT is to limit the maximum pairs to verify, default to be exploiting all possible pairs.
    Positive results are stored as <STU>_0.npy and negative as <STU>_1.npy.
    """
    bins = [[], []]
    if stu:
        for finger1, finger2 in itertools.combinations_with_replacement(list(range(1, 5)), 2):
            for t1, t2 in itertools.combinations_with_replacement(list(range(1, 11)), 2):
                for m1, m2 in [(1, 1)]:
                    path1 = DATA_PATH.format(m1, stu, stu, finger1, t1)
                    path2 = DATA_PATH.format(m2, stu, stu, finger2, t2)
                    if path1 == path2: continue
                    try:
                        ratio = proc(path1, path2)
                    except Exception as e:
                        print(path1, path2, e)
                        ratio = 0
                    if finger1 == finger2: bins[0].append(ratio)
                    else: bins[1].append(ratio)
        bins = [np.array(a) for a in bins]
        np.save(str(stu) + '_0', bins[0])
        np.save(str(stu) + '_1', bins[1])
    else:
        students = os.listdir('class1/ep1/')
        count = 0
        for stu1, stu2 in itertools.combinations_with_replacement(students, 2):
            for finger1, finger2 in itertools.combinations_with_replacement(list(range(1, 5)), 2):
                for t1, t2 in itertools.combinations_with_replacement(list(range(1, 11)), 2):
                    for m1, m2 in [(1, 1)]:
                        path1 = DATA_PATH.format(m1, stu1, stu1, finger1, t1)
                        path2 = DATA_PATH.format(m2, stu2, stu2, finger2, t2)
                        if path1 == path2: continue
                        if pos_only and (finger1 != finger2 or stu1 != stu2): continue
                        elif neg_only and (finger1 == finger2 and stu1 == stu2): continue
                        try:
                            ratio = proc(path1, path2)
                            print(path1, path2, ratio)
                        except ValueError as e:
                            continue
                        count += 1
                        if limit and count > limit: 
                            if not neg_only: np.save('all_0', bins[0])
                            if not pos_only: np.save('all_1', bins[1])
                            return
                        if finger1 == finger2 and stu1 == stu2: bins[0].append(ratio)
                        else: bins[1].append(ratio)
        bins = [np.array(a) for a in bins]
        if not neg_only: np.save('all_0', bins[0])
        if not pos_only: np.save('all_1', bins[1])
        
def plot_stat(stu=None):
    stu = stu or 'all'
    corr = np.load(str(stu) + '_0.npy') * 100
    print(corr.shape, corr[corr == 0].shape)
    #corr = corr[corr > 5]
    incorr = np.load(str(stu) + '_1.npy') * 100
    print(incorr.shape, incorr[incorr == 0].shape)
    #incorr = incorr[incorr > 5]
    sns.distplot(corr, hist=True, kde=True, 
             bins=10, color='g', label='genuine',
             kde_kws={'linewidth': 2})  
    sns.distplot(incorr, hist=True, kde=True, 
             bins=10, color='r', label='imposter',
             kde_kws={'linewidth': 2})
    plt.legend(loc='upper center')
    plt.xlabel('Matched Pixel Ratio(MPR)(%)')
    plt.ylabel('MPR Histogram')
    plt.savefig(str(stu)+'.jpg')
    
def cls_thresh(stu=None):
    """
    Plotting the curve for sensitivity of threshold variance for different statistics.
    """
    stu = stu or 'all'
    corr = np.load(str(stu) + '_0.npy')
    incorr = np.load(str(stu) + '_1.npy')
    thresholds = np.linspace(0, 1, 100)
    def acc(thr, corr, incorr):
        tp = corr[corr > thr].shape[0] + incorr[incorr < thr].shape[0]
        return tp / (corr.shape[0] + incorr.shape[0])
    accs = np.array([acc(x, corr, incorr) for x in thresholds])
    plt.plot(thresholds, accs, 'g.-', label='surf')
    netcorr = np.load('corrects.npy')
    netincorr = np.load('incorrects.npy')
    netaccs = np.array([acc(x, netcorr, netincorr) for x in thresholds])
    plt.plot(thresholds, netaccs, 'r.-', label='relation network')
    plt.legend(loc='lower right')
    plt.xlabel('Thresholds')
    plt.ylabel('Classification Accuracy')
    plt.savefig('thresh-dist.jpg')
    
if __name__ == '__main__':
    #calc_stat(535, proc=sift_process)
    #calc_stat(proc=sift_process, limit=500, neg_only=True)
    #plot_stat()
    #cls_thresh()
    
    path1 = DATA_PATH.format(1, 535, 535, 1, 10)
    path2 = DATA_PATH.format(1, 535, 535, 1, 1)
    path3 = DATA_PATH.format(1, 535, 535, 2, 1)
    
    roi1 = roi.process(path1)
    roi2 = roi.process(path2)
    roi3 = roi.process(path3)
    #print(sift_match(roi1, roi3, thresh=0.75, vis=True))
    sift_match(roi1, roi2, thresh=0.75, vis=True)
    #segm_match(roi1, roi3, vis=True)
