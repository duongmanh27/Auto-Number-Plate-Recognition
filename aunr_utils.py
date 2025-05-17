import cv2 as cv
import numpy as np


def change_contrast(image) :
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    l_image = cv.merge((cl, a, b))
    final_image = cv.cvtColor(l_image, cv.COLOR_LAB2BGR)
    return final_image


def rotate_image(image, angle) :
    image_center = tuple(np.array(image.shape[1 : :-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    results = cv.warpAffine(image, rot_mat, image.shape[1 : :-1], flags=cv.INTER_LINEAR)
    return results


def compute_skew(image, center_thres) :
    if image is None or image.size == 0 :
        print("Warning: Image is empty or None in compute_skew.")
        return 0.0  # Không xoay

    if len(image.shape) == 3 :
        h, w, _ = image.shape
    elif len(image.shape) == 2 :
        h, w = image.shape
    else :
        raise ValueError("unsupported image type")

    image = cv.medianBlur(image, 3)
    edges = cv.Canny(image, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=w / 1.5, maxLineGap=h / 3.0)
    if lines is None :
        return 0  # Không cần xoay nếu không tìm thấy đường thẳng nào

    min_line = 100
    min_line_pos = 0
    for i in range(len(lines)) :
        for x1, y1, x2, y2 in lines[i] :
            center_point = [((x1 + x2) / 2), ((y1 + y2) / 2)]
            if center_thres == 1 and center_point[1] < 7 :
                continue
            if center_point[1] < min_line :
                min_line = center_point[1]
                min_line_pos = i
    angle = 0.0
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos] :
        ang = np.arctan2(y2 - y1, x2 - x1)
        if np.fabs(ang) < 30 :
            angle += ang
            cnt += 1
    if cnt == 0 :
        return 0.0
    return (angle / cnt) * 180 / np.pi


def deskew(image, change_cons, center_thres) :
    if image is None or image.size == 0 :
        print("Warning: Image is empty or None in deskew.")
        return image

    if change_cons == 1 :
        return rotate_image(image, compute_skew(change_contrast(image), center_thres))
    else :
        return rotate_image(image, compute_skew(image, center_thres))
