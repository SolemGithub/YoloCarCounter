import torch.cuda
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


print(torch.cuda.is_available())
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../Videos/cars.mp4")
cap.set(3,1280)
cap.set(4,720)
model = YOLO("../YoloWeights/yolov8l.pt")

# slow or speed
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

# FUN
def contains(newClassNames):
    return currentClass in newClassNames


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]



mask = cv2.imread("mask4.png")
tracker = Sort(max_age = 20, min_hits=2, iou_threshold=0.3)

limits = [350, 297,673,297]
totalCounts = []
prev_frame_time = 0
new_frame_time = 0
searchArea = 5

while True:
    success, img = cap.read()

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, w, h = box.xywh[0]
            w, h = x2 - x1, y2 - y1
            bbox = int(x1), int(y1), int(w), int(h)

            # x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            conf = math.floor((box.conf[0] * 100)) / 100
            print(conf)
            cls = int(box.cls[0])
            currentClass = classNames[cls]


            if contains(["car","motorbike","truck"]) and conf > 0.5:
               # cvzone.putTextRect(img, f'{classNames[cls]} {conf} ', (max(0, (int(x1))), max(40, int(y1 - 40))),
                #                   scale=0.6, offset=5, thickness=1)
               #
               cvzone.cornerRect(img, bbox, l=8, rt=5)
               currentArray = np.array([x1.cpu(),y1.cpu(),x2.cpu(),y2.cpu(),conf])
               detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    for results in resultsTracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2),int(id)
        print(results)
        print(x1)
        #cvzone.cornerRect(img, bbox, l=8, rt=2, colorR=(0,255,255))
        cvzone.putTextRect(img, f'{id}  ', (max(0, (int(x1))), max(40, int(y1 - 40))),
                           scale=2, offset=10, thickness=3)
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(int(cx),int(cy)),3,(255,0,255),cv2.FILLED)

   # cvzone.cornerRect(img, (limits[0], limits[1], limits[2]-limits[0], limits[1]-limits[3]+searchArea), l=8, rt=2, colorR=(0, 255, 255))

        if limits[0] < cx < limits[2] and limits[1]-searchArea < limits[3]+searchArea:

            if totalCounts.count(id) == 0:
                totalCounts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), searchArea)
                cv2.circle(img, (int(cx), int(cy)), 20, (0, 255, 255), cv2.FILLED)

        cvzone.putTextRect(img, f'{len(totalCounts)}  ', (50, 50))

    #cv2.imshow("ImageRegion", imgRegion)p
    cv2.imshow("Image", img)


    key = cv2.waitKey(1)
    if key == ord('p'):
        print("P")
        a = cv2.waitKey(-1)
