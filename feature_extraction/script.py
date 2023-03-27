import cv2
import numpy as np
import math
from collections import defaultdict

image = cv2.imread("sample/paddy_seed.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 9, 25)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

dilate = cv2.dilate(edged, kernel, iterations=1)

# find contours in the dilated image
contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


def get_contour_areas(contours):

    all_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas


sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)


largest_item = sorted_contours[0]
x, y, w, h = cv2.boundingRect(largest_item)
aspect_ratio = float(w)/h

image_copy = image.copy()

# contours on a copy of the original image
cv2.drawContours(image_copy, largest_item, -1, (0, 255, 0), 7)
# cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
text = f"aspect_ratio={aspect_ratio}"
cv2.putText(image_copy, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 5)

cv2.imshow("contours", image_copy)
cv2.waitKey(0)
