import torch
import os
from read_lpn import read_plate, get_car
from aunr_utils import deskew
import cv2 as cv
from sort.sort import Sort
from ultralytics import YOLO
import numpy as np
from pprint import pprint


video_path = "duong_thuong.mp4"
folder_path = "images"

vehicle = [2, 3, 5, 7]
# load model
object_detector = YOLO("yolov8n.pt")
yolo_LP_detect = torch.hub.load("yolov5", 'custom', path='models/model_detectd_trained.pt',
    force_reload=True,
    source='local')
yolo_license_plate = torch.hub.load("yolov5", 'custom', path='models/model_ocr_trained.pt',
    force_reload=True,
    source='local')
tracker = Sort(max_age=30)
yolo_license_plate.conf = 0.60
car_detections = {}
results = {}
best_results = {}
batch_results = {}
prev_active_ids = set()
saved_car_ids = set()
cap = cv.VideoCapture(video_path)
frame_nmr = -1
if not os.path.isdir(folder_path) :
    os.mkdir(folder_path)
while cap.isOpened() and frame_nmr < 6000 :
    frame_nmr += 1
    if frame_nmr % 10 == 0 :
        batch_results = {}
    flag, frame = cap.read()
    if not flag :
        break
    text = ''
    results[frame_nmr] = {}
    current_active_ids = set()
    best_results[frame_nmr] = {}
    detections_ = []
    detection_vehicle = yolo_LP_detect(frame, size=max(frame.shape[0], frame.shape[1]))
    detections = object_detector(frame)
    detections_ = []
    for detection in detections[0].boxes.data.tolist() :
        x_car1, y_car1, x_car2, y_car2, conf_car, cls = detection

        cv.rectangle(frame, (int(x_car1), int(y_car1)), (int(x_car2), int(y_car2)), (0, 0, 255), 1)
        if int(cls) in vehicle :
            detections_.append([x_car1, y_car1, x_car2, y_car2, conf_car])
    if len(detections_) == 0 :
        detection_np = np.empty((0, 5))
    else :
        detection_np = np.asarray(detections_)
    track_ids = tracker.update(detection_np)
    for detection in detection_vehicle.pandas().xyxy[0].values.tolist() :
        x1 = int(detection[0])
        y1 = int(detection[1])
        x2 = int(detection[2])
        y2 = int(detection[3])
        license_plate = detection[:6]
        x_car1, y_car1, x_car2, y_car2, car_id = get_car(license_plate, track_ids)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if car_id != -1 :
            current_active_ids.add(car_id)
            license_plate_crop = frame[y1 :y2, x1 :x2]
            vehicle_crop = frame[int(y_car1) :int(y_car2), int(x_car1) :int(x_car2)]
            for cc in range(2) :
                for ct in range(2) :
                    result = read_plate(yolo_license_plate, deskew(license_plate_crop, cc, ct))

                    if isinstance(result, tuple) and len(result) == 2 :
                        text, text_score = result
                    else :
                        print("Error: read_plate did not return exactly 2 values.")
                        text, text_score = None, None  # Hoặc gán giá trị mặc định
                    if text != "unknown" or text is not None :
                        results[frame_nmr][car_id] = {'vehicle' : {'bbox' : [x_car1, y_car1, x_car2, y_car2]},
                                                      'license_plate' : {'bbox' : [x1, y1, x2, y2],
                                                                         'text' : text,
                                                                         'text_score' : text_score
                                                                         }
                                                      }

                    batch_results[car_id] = {
                        'vehicle' : {'bbox' : [x_car1, y_car1, x_car2, y_car2]},
                        'license_plate' : {'bbox' : [x1, y1, x2, y2],
                                           'text' : text,
                                           'text_score' : text_score
                                           },
                        'plate_crop' : license_plate_crop,
                        'vehicle_crop' : vehicle_crop
                    }

                    cv.putText(frame, text, (int(x_car1 + 120), int(y_car1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2)
        cv.putText(frame, f"ID: {car_id}", (int(x_car1), int(y_car1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
            2)
    if frame_nmr > 0 :
        lost_ids = prev_active_ids - current_active_ids
        for lost in lost_ids :
            if lost in best_results and lost not in saved_car_ids :
                vehicle_crop = best_results[lost].get('vehicle_crop', None)
                if vehicle_crop is not None :
                    best_text = best_results[lost]['license_plate']['text']
                    filename = os.path.join(folder_path, "{}.jpg".format(best_text))
                    cv.imwrite(filename, vehicle_crop)
                    saved_car_ids.add(lost)
                    print("Save car image of car_id {} to file {}".format(lost, filename))
    if frame_nmr % 10 == 9 :
        for cid, detection in batch_results.items() :
            new_score = detection['license_plate']['text_score']
            if cid not in best_results :
                best_results[cid] = detection
                best_text = detection['license_plate']['text']
                filename = os.path.join(folder_path, f"{best_text}.jpg")
                cv.imwrite(filename, detection['plate_crop'])
            else :
                existing_score = best_results[cid].get('license_plate', {}).get('text_score')
                if existing_score is None :
                    best_results[cid] = detection
                    best_text = detection['license_plate']['text']
                    filename = os.path.join(folder_path, f"{best_text}.jpg")
                    cv.imwrite(filename, detection['plate_crop'])
                elif new_score is not None and existing_score is not None and new_score > existing_score :
                    best_results[cid] = detection
                    best_text = detection['license_plate']['text']
                    filename = os.path.join(folder_path, f"{best_text}.jpg")
                    cv.imwrite(filename, detection['plate_crop'])

    cv.imshow("frame", frame)
    prev_active_ids = current_active_ids.copy()
    if cv.waitKey(1) & 0xFF == ord(' ') :
        break
pprint(best_results)
cap.release()
cv.destroyAllWindows()
