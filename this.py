#first
import cv2
import numpy as np
def get_thresh(image):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
 
    cv2.imshow("help1", img)
    cv2.waitKey(0)
    (h1, l1, s1) = (0, 163, 0)
    (h2, l2, s2) = (255, 255, 255)
 
    h_min = np.array((h1, l1, s1), np.uint8)
    h_max = np.array((h2, l2, s2), np.uint8)
    return cv2.inRange(hls, h_min, h_max)
def transform_thresh(thresh):
    res = cv2.bitwise_and(img, img, mask = thresh)
    cv2.imshow("help2", thresh)
    cv2.waitKey(0)
    cv2.imshow("help3", res)
    cv2.waitKey(0)
    shape = (img.shape[0], img.shape[1], 3)
    black = np.zeros(shape, np.uint8)
    black1 = cv2.rectangle(black, (0, 250), (800, 667), (255, 255, 255), -1)
    gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
    ret, b_mask = cv2.threshold(gray, 127, 255, 0)
    fin = cv2.bitwise_and(res, res, mask = b_mask)
 
    cv2.imshow("res.jpg", fin)
 
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(fin, low_threshold, high_threshold)
    return cv2.GaussianBlur(edges, (3, 3), 0)
 
def get_final_image(edges, image):
    line_image = np.copy(image) * 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, np.array([]), 50, 5)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
 
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
 
    cv2.imshow('result', lines_edges)
    cv2.imwrite('img_out.jpg', lines_edges)
 
    cv2.waitKey(0)
 

img = cv2.imread("1.jpg")
print(img.shape)
thresh = get_thresh(img)
edges = transform_thresh(thresh)
get_final_image(edges, img)
    
cv2.destroyAllWindows()
print("Complete")
 
 
#second
import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import os
 
def get_thresh(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    (h1, l1, s1) = (0, 163, 0)
    (h2, l2, s2) = (255, 255, 255)
 
    h_min = np.array((h1, l1, s1), np.uint8)
    h_max = np.array((h2, l2, s2), np.uint8)
    return cv2.inRange(hls, h_min, h_max)
 
def transform_thresh(thresh):
    res = cv2.bitwise_and(img, img, mask = thresh)
    black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    black1 = cv2.rectangle(black, (430, 500), (1024, 739), (255, 255, 255), -1)
    gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
    ret, b_mask = cv2.threshold(gray, 127, 255, 0)
    fin = cv2.bitwise_and(res, res, mask = b_mask)
    edges = cv2.Canny(fin, 50, 150)
    return cv2.GaussianBlur(edges, (3, 3), 0)
 
def get_final_video(edges, image):
    line_image = np.copy(image) * 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, np.array([]), 50, 5)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
 
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    cv2.imshow('result', lines_edges)
    videoWriter.write(lines_edges)
    return cv2.waitKey(1) == ord('q')
 
videoPath = os.getcwd() + '/video.mp4'
videoCapture = cv2.VideoCapture(videoPath)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(3)),
        int(videoCapture.get(4)))
videoWriter = cv2.VideoWriter('new_video_out.avi',
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
 
while True:
    is_true, img = videoCapture.read()
    if is_true:
        thresh = get_thresh(img)
        edges = transform_thresh(thresh)
        if get_final_video(edges, img):
            break
    else:
        exit("OK")
 
videoCapture.release()
VideoWriter.release()
cv2.destroyAllWindows()
print("OK")