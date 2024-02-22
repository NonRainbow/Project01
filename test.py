import numpy as np
import cv2 as cv
import time

cap = cv.VideoCapture("videos/big.mp4")
paraPath = "Camera.yaml"
dilate_core = np.ones((12,12),np.uint8)
erode_core = np.ones((3,3),np.uint8)
eps = 0.08
scale = 1
min_center_area = 800
max_center_area = 1400
center_coords = (0,0)
radius = 160
outline_radius = 300
p_min_area =10000
panel_centers = []
min_light_area = 10
hit_point = (0,0)
hit_point_detect = 50
hit_dilate_core = np.ones((hit_point_detect,hit_point_detect),np.uint8)

is_opened = cap.isOpened() #check wether the video is correctly opened

while is_opened:
    #time.sleep(0.03)

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly then ret is True

    if not ret:
        print("Can't receive frame . Exiting ...")
        cv.waitKey()
        break

    # Our operations on the frame come here
    b,g,r = cv.split(frame)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #blur = cv.GaussianBlur(b,(3,3),1.3)
    ret, thres = cv.threshold(b,176,255,cv.THRESH_BINARY)
    #erode = cv.erode(thres,erode_core,2)         #erode
    dilate = cv.dilate(thres,dilate_core,6)        #dilate
    
    contours,hie = cv.findContours(dilate,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    res1 = frame.copy()
    res2 = frame.copy()
    
    bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]
    for bbox in bounding_boxes:
        [x,y,w,h] = bbox
        if (w/h < scale + eps and w/h > scale - eps) and (w * h < max_center_area and w * h > min_center_area):
            cv.rectangle(res2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_coords = (int(x + 0.5 * w),int(y + 0.5 * h)) #get center R

    out_panels = dilate.copy()
    cv.circle(out_panels,center_coords,radius,(0,0,0),-1) #draw solid circle to hide strips
    #cv.circle(out_panels,center_coords,outline_radius,(0,0,0),10) #draw outer circle line to remove influences
    #cv.circle(out_panels,center_coords,outline_radius,(0,255,0),10) #test
    
    contours_out,hie_out = cv.findContours(out_panels,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    bounding_boxes_p = [cv.boundingRect(cnt) for cnt in contours_out]
    
    for bbox in bounding_boxes_p:
        [x,y,w,h] = bbox
        if w * h > p_min_area :
            cv.rectangle(res2, (x, y), (x + w, y + h), (0, 255, 225), 2)
            temp_center = (int(x + 0.5 * w),int(y + 0.5 * h))
            panel_centers.append(temp_center) #get panel centers
            cv.circle(res2,temp_center,2,(0,255,255),2)
            
            hit_roi = out_panels[(temp_center[0] - hit_point_detect // 2):(temp_center[0] + hit_point_detect),(temp_center[1] - hit_point_detect // 2):(temp_center[1] + hit_point_detect)]
            #get center rough area
            
            if np.mean(hit_roi) > 0 :
                hit_point = temp_center
    
    cv.circle(res2,hit_point,10,(0,0,255),2)
    
    
    # Display the resulting frame
    #cv.imshow("frame", frame)
    #cv.imshow("blur", blur)
    #cv.imshow("thres", thres)
    cv.imshow("dilate", dilate)
    cv.imshow("boxes", res2)
    cv.imshow("panels", out_panels)
    

    #output data
    #print (center_coords)
    
    #clear former data
    panel_centers = []

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()