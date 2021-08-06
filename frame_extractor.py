import cv2 

cap = cv2.VideoCapture('depth_left.avi')
i = 0

print('Extracting frames')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame = cv2.resize(frame,(640,480))
    cv2.imwrite('./cal_left/frame_left_' + str(i) + '.png', frame)
    i += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()