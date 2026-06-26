#書類検出


import cv2

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
    cv2.drawContours(result, [document], -1, (0, 255, 0), 3)

cv2.imshow("Document Detection", result)

cv2.waitKey(0)

cv2.destroyAllWindows()