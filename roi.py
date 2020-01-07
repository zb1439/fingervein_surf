"""
Author: Zhibo Fan
Extracting Finger Vein RoI for post-processing 
(e.g., segmentation, keypoint matching, deep learning)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser('fingerdata/class[class]/[index]/[finger]/[machine]/[which].jpg')
parser.add_option('-i', '--index', dest='index')
parser.add_option('-m', '--machine', dest='machine')
parser.add_option('-f', '--finger', dest='finger')
parser.add_option('-w', '--which', dest='which')
options, args = parser.parse_args()

index = options.index or '535'
finger = 1 if not options.finger else int(options.finger)
assert finger in [1, 2, 3, 4]
machine = 1 if options.machine is None else int(options.machine)
assert machine in [0, 1]
which = 2 if options.which is None else int(options.which)
assert which in list(range(1, 11))

# Change this DATA_PATH to your custom path format!
DATA_PATH = 'class1/ep{}/{}/{}-{}-{}-1.bmp'.format(machine, index, index, finger, which)
    
def clahe(src, clip=15, vis=False, gray=False):
    afterproc = src.copy()
    clahe = cv2.createCLAHE(clip, tileGridSize=(8, 8))
    afterproc = np.transpose(afterproc, [2, 0, 1])
    afterproc = [clahe.apply(ch) for ch in afterproc]
    afterproc = np.transpose(np.array(afterproc), [1, 2, 0])
    if gray:
        afterproc = cv2.cvtColor(afterproc, cv2.COLOR_BGR2GRAY)
    if vis:
        cv2.imwrite('clahe.jpg', afterproc)
    return afterproc
    
def equalHist(src, vis=False, gray=False):
    afterproc = src.copy()
    afterproc = np.transpose(afterproc, [2, 0, 1])
    afterproc = [cv2.equalizeHist(ch) for ch in afterproc]
    afterproc = np.transpose(np.array(afterproc), [1, 2, 0])
    if gray:
        afterproc = cv2.cvtColor(afterproc, cv2.COLOR_BGR2GRAY)
    if vis:
        cv2.imwrite('clahe.jpg', afterproc)
    return afterproc

def max_ctr(src, vis=False):
    """ Passing the filtered image (without CLAHE),
    returns the longest contour of its otsu segmentation in shape [N, 2].
    """
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, 3)
    
    thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    binary = binary.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) > 0, "No contours found!"
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    maxctr = contours[-1]
    #maxctr = max(contours, key=lambda x: x.shape[0])
    if vis:
        _src = src.copy()
        cv2.drawContours(_src, [maxctr], -1, (255, 0, 0), 3)
        cv2.imwrite('max-ctr.jpg', _src)
        cv2.imwrite('max-ctr-binary.jpg', binary)
    return maxctr[:, 0, :], gray
    
def extract_bbox(src, ctr, vis=False):
    """ Extract bounding boxes given a contour and updated contour. """
    is_src_list = isinstance(src, list)
    if not is_src_list: 
        src = [src]
    miny = np.min(ctr[:, 1])
    maxy = np.max(ctr[:, 1])
    new_ctr = ctr - np.array([[0, miny]])
    new_src = []
    for img in src:
        new_img = img[miny:maxy, :]
        new_src.append(new_img)
    if vis:
        cv2.imwrite('extract-bbox.jpg', src[0])
    if not is_src_list:
        new_src = new_src[0]
    return new_src, new_ctr
    
def extract_ibox(src, vis=False):
    """ Extract inner most box. """
    height = src.shape[0]
    margin = 50
    new_src = src[:, 200:-50, :]
    img_canny = cv2.Canny(new_src, 90, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_canny = cv2.dilate(img_canny, kernel)
    lines = cv2.HoughLinesP(img_canny, 1, np.pi / 180, 150, minLineLength=50, maxLineGap=20)[:, 0, :]
    points = np.concatenate(np.split(lines, 2, axis=-1), axis=0)
    upbound = np.max(points[points[:, 1] < height // 2 - margin, 1])
    downbound = np.min(points[points[:, 1] > height // 2 + margin, 1])
    crop = src[upbound:downbound, :]
    if vis:
        cv2.imwrite('extract-ibox.jpg', crop)
    return crop
    
def find_bounding_lines(src, ctr, vis=False):
    """ Returns a affine matrix to make the midline straight. """
    new_src = src[:, 100:-100, :]
    img_canny = cv2.Canny(new_src, 90, 150)
    kernel = kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_canny = cv2.dilate(img_canny, kernel)
    lines = cv2.HoughLinesP(img_canny, 1, np.pi / 180, 150, minLineLength=50, maxLineGap=20)[:, 0, :]
    # Grouping two lines
    uplines = lines[lines[:, 1] < 100]
    uplines = uplines[uplines[:, 3] < 100]
    downlines = lines[lines[:, 1] > 200]
    downlines = downlines[downlines[:, 3] > 200]
    upleftp = uplines[np.argmin(uplines[:, 0]), :2]
    uprightp = uplines[np.argmax(uplines[:, 2]), 2:]
    downleftp = downlines[np.argmin(downlines[:, 0]), :2]
    downrightp = downlines[np.argmax(downlines[:, 2]), 2:]
    midleftp = (upleftp + downleftp) // 2
    midrightp = (uprightp + downrightp) // 2
    # Get affine matrix
    k = (midleftp[1] - midrightp[1]) / (midleftp[0] - midrightp[0])
    theta = -np.arctan(k)
    M = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]])
    if vis:
        _src = src.copy()
        bias = np.array([100, 0])
        cv2.imwrite('canny.jpg', img_canny)
        cv2.line(_src, tuple(upleftp+bias), tuple(uprightp+bias), (255, 0, 0), 2)
        cv2.line(_src, tuple(downleftp+bias), tuple(downrightp+bias), (255, 0, 0), 2)
        cv2.line(_src, tuple(midleftp+bias), tuple(midrightp+bias), (255, 0, 0), 2)
        cv2.imwrite('find-bounding-lines.jpg', _src)
    return M
    
def get_center_roi(src, vis=False):
    """ Returns centered roi only. """
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    bright = gray.sum(0)
    brightmost = np.argmax(bright)
    bright = np.concatenate([bright[:brightmost], bright[brightmost+1:]])
    def deweight(x, center, coeff=0.01):
        x = center - x if x < center else x + 1 - center
        x *= coeff
        return 1 / (1 + np.exp(-x))
        
    bright = np.array([x * deweight(i, brightmost) for i, x in enumerate(bright)])
    brightsecond = np.argmax(bright)
    if brightsecond >= brightmost: brightsecond += 1
    if vis:
        _src = src.copy()
        cv2.line(_src, (brightmost, 0), (brightmost, src.shape[0] - 1), (255, 0, 0), 2)
        cv2.line(_src, (brightsecond, 0), (brightsecond, src.shape[0] - 1), (255, 0, 0), 2)
        cv2.imwrite('get-center-roi.jpg', _src)
    return src[:, min(brightmost, brightsecond):max(brightmost, brightsecond)]

def process(path):
    src = cv2.imread(path)
    src = cv2.medianBlur(src, 5)
    src = cv2.GaussianBlur(src, (5, 5), 0)
    ctr, gray = max_ctr(src)
    (bbox, gray), ctr = extract_bbox([src, gray], ctr)
    M = find_bounding_lines(bbox, ctr)
    rotate = cv2.warpAffine(src, M, src.shape[:2][::-1])
    ctr, gray = max_ctr(rotate)
    (crop, gray), ctr = extract_bbox([rotate, gray], ctr)
    crop = extract_ibox(crop)
    roi = get_center_roi(crop) 
    return equalHist(roi)
    
if __name__ == '__main__':
    src = cv2.imread(DATA_PATH)
    src = cv2.medianBlur(src, 5)
    src = cv2.GaussianBlur(src, (5, 5), 0)
    ctr, gray = max_ctr(src, True)
    (bbox, gray), ctr = extract_bbox([src, gray], ctr, True)
    M = find_bounding_lines(bbox, ctr, vis=True)
    rotate = cv2.warpAffine(src, M, src.shape[:2][::-1])
    ctr, gray = max_ctr(rotate, False)
    crop = extract_ibox(crop, True)
    roi = get_center_roi(crop, True)
