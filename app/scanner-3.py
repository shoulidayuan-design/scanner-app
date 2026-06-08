#透視変換


import cv2
import numpy as np

def order_points(pts):

    pts = pts.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)] #左上
    rect[2] = pts[np.argmax(s)] #右下

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)] #右上
    rect[3] = pts[np.argmax(diff)] #左下

    return rect

def four_point_transform(image, pts):

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB)) #幅の最大値

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(int(heightA), int(heightB)) #高さの最大値

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32") #変換後の座標

    M = cv2.getPerspectiveTransform(rect, dst) #変換行列

    warped = cv2.warpPerspective(
        image,
        M,
        (maxWidth, maxHeight)
    ) #透視変換

    return warped

img = cv2.imread("document.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

edge = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(
    edge,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

document = None
max_area = 0

for c in contours:

    area = cv2.contourArea(c)

    if area < 5000:
        continue

    peri = cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:

        if area > max_area:
            document = approx
            max_area = area

result = img.copy()

if document is not None:

    cv2.drawContours(
        result,
        [document],
        -1,
        (0, 255, 0),
        3
    )

    warped = four_point_transform(
        img,
        document
    )

    cv2.imshow(
        "Scanned Document",
        warped
    )

else:
    print("書類が検出できませんでした。")

cv2.imshow(
    "Document Detection",
    result
)

cv2.waitKey(0)

cv2.destroyAllWindows()