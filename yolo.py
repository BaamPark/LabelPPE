from ultralytics import YOLO
import cv2
import yaml

def run_yolo(source):
    model = YOLO('18cls_ppe_detector.pt')
    results = model(source)
    frame = results[0].orig_img
    bbox_list = []

    clsnum_to_name = import_yaml(True)
    for result in results:
        for box in result.boxes:
            print("person", box.xyxy[0].tolist())
            bbox = [int(x) for x in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = bbox
            top_left = (x1, y1)
            bottom_right = (x2, y2)

            cv2.rectangle(frame, top_left, bottom_right, color=(0, 255, 0), thickness=2)
                
                #for mot format
            bb_left = x1
            bb_top = y1
            bb_width = x2 - x1
            bb_height = y2 - y1

            bbox_list.append([bb_left, bb_top, bb_width, bb_height, clsnum_to_name[int(box.cls.item())]])

    return frame, bbox_list

def import_yaml(flag_key_number=True):
    with open('yolo_cls_config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    if flag_key_number == False:
        #swap key and value
        data = {value: key for key, value in data.items()}

    return data
