import matplotlib.pylab as plt
import cv2
import numpy as np


def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])  # bottom of the image

    y2 = int(y1 * 3 / 5)  # slightly lower than the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -0.1:
                left_fit.append((slope, intercept))
            elif slope > 0.1:
                right_fit.append((slope, intercept))
            else:
                return None


    left_fit_average = np.average(left_fit, axis=0)
    print('left-fit: {}'.format(left_fit_average))
    right_fit_average = np.average(right_fit, axis=0)
    print('right-fit: {}'.format(right_fit_average))
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    # print(averaged_lines)
    return averaged_lines

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if(lines is not None):
        for line in lines:
            # print(line)
            for x1, y1, x2, y2 in line:
                # print(x1)
                # print(y1)
                # print(x2)
                # print(y2)
                cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def process(image):
    # print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 8 * 3, height / 8 * 5),
        (width / 8 * 5, height / 8 * 5),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 300, 320)
    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    averaged_lines = average_slope_intercept(frame, lines)
    image_with_lines = draw_the_lines(image, averaged_lines)
    #image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

def birdseye(image):
    IMAGE_H = 270
    IMAGE_W = 1280
    src = np.float32([[0, IMAGE_H], [1280, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[500, IMAGE_H], [720, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation


    image = image[450:(450 + IMAGE_H), 0:IMAGE_W]  # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))
    return warped_img

cap = cv2.VideoCapture('vid1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output4.mp4', fourcc, 20.0, (1280, 720))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame,(1280,720))
        frame = birdseye(frame)
        frame = process(frame)
        cv2.imshow('frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()