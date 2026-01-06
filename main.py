# -*- coding: utf-8 -*-
# @Time    : 2025/7/9 10:02
# @Author  : Chaohe Wen
# @Email   : wenchaohe@tsingtec.com

import os
import cv2
import algo
import json
import dotenv
import imutils
import argparse
from algo.util import calculate_iou

dotenv.load_dotenv()

YOLO_MODEL = os.getenv("YOLO_MODEL")
FIGHT_MODEL = os.getenv("FIGHT_MODEL")
CIGAR_MODEL = os.getenv("CIGAR_MODEL")
FACE_MODEL = os.getenv("FACE_MODEL")
PACK_MODEL = os.getenv("PACK_MODEL")

class CallbackRES(object):
    format_dict = {
        "frame_id": 1, 							#视频帧
        "ori_img": None,                        #原始图像
        "face_det":	[],                         #人脸左上角、右下角
        "person_det": [], 					    #人体左上角、右下角
        "brawl": str(False),      				#打架斗殴
        "smoking": str(False),					#抽烟
        "violent_sorting": str(False),  		#暴力分拣
        "intrusion_det": []               		#入侵区域检测, 不支持
        }

def main(video_input):
    FIGHT_ON = False
    FIGHT_ON_TIMEOUT = 5  # second

    fdet = algo.FightDetector(FIGHT_MODEL)
    yolo = algo.YoloPoseEstimation(YOLO_MODEL)
    cigar_det = algo.CigaretteDetector(CIGAR_MODEL)
    face_det = algo.FaceDetector(FACE_MODEL)
    pack_det = algo.PackageDetector(PACK_MODEL)
    act_det = algo.ViolenceDetector()
    cap = cv2.VideoCapture(video_input)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 1 
    results_list = list()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # if frame_idx % 5 == 0:
        if frame_idx % 20 == 0:
            print(f"Processing frame {frame_idx}/{count}")
        person_res = yolo.infer(frame) 
        pack_res = pack_det.infer(frame)
        person_act = act_det.process_frame(person_res)
        # Get the result image from YOLOv8
        result_frame = person_res.plot()
        frame_height = result_frame.shape[0]
        frame_width = result_frame.shape[1]
        if result_frame.shape[0] > 720:
            result_frame = imutils.resize(result_frame, width=1280)
        try:
            boxes = person_res.boxes.xyxy.tolist()
            xyn = person_res.keypoints.xyn.tolist()
            confs = person_res.keypoints.conf
            ids = person_res.boxes.id   

            confs = [] if confs is None else confs.tolist()
            ids = [] if ids is None else [str(int(ID)) for ID in ids]

            for person_box in boxes:
                x1, y1, x2, y2 = map(int, person_box)
                person_img = frame[y1:y2, x1:x2]
                cur_frame_res = CallbackRES.format_dict.copy()
                cur_frame_res["frame_id"] = frame_idx
                cur_frame_res["person_det"] = person_box
                # second action detection
                if len(person_act["bbox"]) > 0 and person_act["bbox"] == person_box:
                    # iou
                    for pack_box in pack_res:
                        if calculate_iou(person_box, pack_box) > 0.2:
                            cur_frame_res["violent_sorting"] = str(True)
                            break

                # smoking detection
                is_person_smoking = cigar_det.infer(person_img)
                cur_frame_res["smoking"] = is_person_smoking
                # face detection
                face_box = face_det.infer(person_img)
                cur_frame_res["face_det"] = face_box
                results_list.append(cur_frame_res)  
            #####################fight detection#####################
            interaction_boxes = algo.get_interaction_box(boxes)

            # Process only what is inside the interaction box
            for inter_box in interaction_boxes:
                # Green box 
                cv2.rectangle(result_frame, (int(inter_box[0]), int(inter_box[1])),
                              (int(inter_box[2]), int(inter_box[3])), (0, 255, 0), 2)

                # Prediction starts here - per person - all person on the frame - including outside the interaction box
                both_fighting = []
                latest = len(boxes)
                for conf, xyn, box, identity in zip(confs, xyn, boxes, ids):
                    # Check if the person is within the interaction box - filter only persons inside the interaction box
                    center_person_x, center_person_y = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
                    if inter_box[0] <= center_person_x <= inter_box[2] and inter_box[1] <= center_person_y <= \
                            inter_box[3]:
                        # Fight Detection
                        is_person_fighting = fdet.detect(conf, xyn)
                        both_fighting.append(is_person_fighting)
                        for r in results_list[-latest:]:
                            if r["person_det"] == box:
                                cur_frame_res["brawl"] = str(True) if is_person_fighting else str(False)
                      
                else:
                    # Check if both fighting
                    if all(both_fighting):
                        FIGHT_ON = True     

        except TypeError as te:
            pass
        except IndexError as ie:
            pass

        # RING THE ALARM 
        if FIGHT_ON:
            FIGHT_ON_TIMEOUT -= 0.2

        if FIGHT_ON_TIMEOUT <= 0:
            FIGHT_ON = False
            FIGHT_ON_TIMEOUT = 5
        frame_idx += 1
        
    with open('output.json', 'w') as file:
        json.dump(results_list, file, indent=4) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video", type=str, 
                        default=r"/home/tsingtec/wen/demo/Action_Recog/data/fight.mp4")
    args = parser.parse_args()
    main(args.video)