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
aspect_ratio = round(float(w)/h, 2)

image_copy = image.copy()

# contours on a copy of the original image
cv2.drawContours(image_copy, largest_item, -1, (0, 255, 0), 7)
# cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
text = f"aspect_ratio={aspect_ratio}"
# cv2.putText(image_copy, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
#                 1.5, (0, 255, 0), 5)

print(text)

for eps in np.linspace(0.001, 0.05, 50):
    # approximate the contour
    peri = cv2.arcLength(largest_item, True)
    approx = cv2.approxPolyDP(largest_item, eps * peri, True)

    # draw the approximated contour on the image
    output = image.copy()
    cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
    text = "eps={:.4f}, num_pts={}, aspect_ratio={}".format(
        eps, len(approx), aspect_ratio)
    cv2.putText(output, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    # Iterate through the vertices of the polygon and mark the coordinates
    for i in range(len(approx)):
        x1, y1 = approx[i][0]
        x2, y2 = approx[(i+1) % len(approx)][0]

        txt = f"({x1},{y1})"

        # Mark the coordinates of each vertex on the image
        cv2.circle(output, (x1 + 15, y1 + 15), 5, (0, 0, 255), -1)
        cv2.putText(output, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Approximated Contour", output)
    cv2.waitKey(0)

cv2.imshow("contours", image_copy)
cv2.waitKey(0)
