#カメラ撮影


import cv2

cap = cv2.VideoCapture(0)

while True:
    
    ret,frame = cap.read()
    
    if not ret:
        break
    
    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1)
    
    if key == ord("s"):
        cv2.imwrite("document.jpg", frame)
        
    elif key == 27:
        break
    
cap.release()

cv2.destroyAllWindows()