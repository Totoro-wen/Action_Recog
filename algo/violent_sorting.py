# -*- coding: utf-8 -*-
# @Time    : 2025/7/11 01:21
# @Author  : Chaohe Wen
# @Email   : wenchaohe@tsingtec.com

import cv2
import time
import math
import numpy as np
from collections import deque
from datetime import datetime

class ViolenceDetector:
    def __init__(self, model=None):
        self.model = model
        # 关键点索引（根据YOLO的17个关键点）
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # 动作分析参数
        self.history = deque(maxlen=10)  # 存储最近10帧的关键点数据
        self.violence_threshold = 0.5  # 暴力行为检测阈值
        self.throw_threshold = 0.7    # 抛掷动作阈值
        self.kick_threshold = 0.6      # 踢踹动作阈值
        self.drop_threshold = 0.5      # 物品掉落阈值
        
        # 报警系统
        self.last_alert_time = 0
        self.alert_cooldown = 5  # 报警冷却时间（秒）
        
        # 可视化参数
        self.colors = {
            "normal": (0, 255, 0),    # 正常 - 绿色
            "throw": (0, 0, 255),     # 抛掷 - 红色
            "kick": (255, 0, 0),      # 踢踹 - 蓝色
            "drop": (255, 255, 0)     # 掉落 - 黄色
        }
    
    def calculate_angle(self, a, b, c):
        """计算三个点之间的角度"""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def calculate_velocity(self, kpt1, kpt2, frame_count):
        """计算关键点运动速度"""
        if not self.history or frame_count < 2:
            return 0
        
        prev_kpts = self.history[-1][0]  # 前一帧的关键点
        if len(prev_kpts) != len(kpt1):
            return 0
        
        # 计算手腕关键点的位移
        wrist_displacement = 0
        wrist_indices = [9, 10]  # 左右手腕的索引
        
        for idx in wrist_indices:
            # if kpt1[idx][2] > 0.5 and prev_kpts[idx][2] > 0.5:  # 置信度>0.5
            dx = kpt1[idx][0] - prev_kpts[idx][0]
            dy = kpt1[idx][1] - prev_kpts[idx][1]
            wrist_displacement += math.sqrt(dx*dx + dy*dy)
        
        # 标准化位移（基于帧数）
        velocity = wrist_displacement / frame_count
        return velocity
    
    def detect_violence(self, keypoints, frame_count):
        """检测暴力分拣行为"""
        results = {"throw": False, "kick": False, "drop": False}
        
        # 需要的关键点索引
        shoulder_idx = [5, 6]    # 左右肩膀
        elbow_idx = [7, 8]       # 左右肘部
        wrist_idx = [9, 10]      # 左右手腕
        hip_idx = [11, 12]       # 左右臀部
        ankle_idx = [15, 16]     # 左右脚踝
        
        # 1. 检测抛掷动作 (手臂运动)
        arm_angles = []
        for side in range(2):  # 左右两侧
            shoulder = keypoints[shoulder_idx[side]]
            elbow = keypoints[elbow_idx[side]]
            wrist = keypoints[wrist_idx[side]]
            
            angle = self.calculate_angle(shoulder[:2], elbow[:2], wrist[:2])
            if not math.isnan(angle):
                arm_angles.append(angle)
        
        # 计算手臂弯曲角度变化
        arm_angle_change = 0
        if len(arm_angles) > 0:
            avg_arm_angle = sum(arm_angles) / len(arm_angles)
            
            # 如果有历史数据
            if self.history and len(self.history) >= 5:
                # 获取5帧前的关键点数据
                prev_kpts = self.history[-5]  # 直接获取关键点数组
                
                prev_angles = []
                for side in range(2):
                    prev_shoulder = prev_kpts[shoulder_idx[side]]
                    prev_elbow = prev_kpts[elbow_idx[side]]
                    prev_wrist = prev_kpts[wrist_idx[side]]
                    
                    angle = self.calculate_angle(
                        prev_shoulder[:2], prev_elbow[:2], prev_wrist[:2]
                    )
                    if not math.isnan(angle):
                        prev_angles.append(angle)
                
                if prev_angles:
                    prev_avg_angle = sum(prev_angles) / len(prev_angles)
                    arm_angle_change = abs(avg_arm_angle - prev_avg_angle)
        
        # 2. 检测踢踹动作 (腿部运动)
        leg_angles = []
        for side in range(2):  # 左右两侧
            hip = keypoints[hip_idx[side]]
            knee = keypoints[13+side]  # 13:left_knee, 14:right_knee
            ankle = keypoints[ankle_idx[side]]
            
            # 只处理置信度较高的关键点
            # if hip[2] > 0.3 and knee[2] > 0.3 and ankle[2] > 0.3:
            angle = self.calculate_angle(hip[:2], knee[:2], ankle[:2])
            if not math.isnan(angle):
                leg_angles.append(angle)
        
        # 3. 计算手腕速度（用于抛掷检测）
        wrist_velocity = self.calculate_velocity(keypoints, keypoints, frame_count)
        
        # 4. 检测物品掉落（手腕高度突然降低）
        drop_detected = False
        wrist_height = 0
        count = 0
        for i in wrist_idx:
            wrist_height += keypoints[i][1]
            count += 1
        if count > 0:
            wrist_height /= count  # 平均高度
        
        if self.history and len(self.history) >= 5:
            prev_kpts = self.history[-5]  # 5帧前的关键点数据
            prev_height = 0
            prev_count = 0
            for i in wrist_idx:
                prev_height += prev_kpts[i][1]
                prev_count += 1
            if prev_count > 0:
                prev_height /= prev_count  # 平均高度
                height_change = abs(wrist_height - prev_height)
                if height_change > self.drop_threshold * 100:  # 阈值调整
                    drop_detected = True
        
        # 决策逻辑
        if arm_angle_change > self.throw_threshold * 30 and wrist_velocity > self.throw_threshold:
            results["throw"] = True
        
        if leg_angles and max(leg_angles) > self.kick_threshold * 100:
            results["kick"] = True
        
        if drop_detected:
            results["drop"] = True
        
        return results
    
    def visualize_results(self, frame, people, violence_results):
        """在视频帧上可视化结果"""
        height, width = frame.shape[:2]
        
        # 显示标题
        cv2.putText(frame, "Violence Detection System", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 绘制检测到的人员
        for person in people:
            # 绘制边界框
            bbox = person["bbox"]
            cv2.rectangle(frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         (0, 255, 0), 2)
            
            # 绘制关键点
            for kp in person["keypoints"]:
                # if kp[2] > 0.3:  # 只绘制置信度>0.3的关键点
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)
            
            # 绘制骨架
            # 躯干
            self.draw_line(frame, person, 5, 6)  # 肩膀
            self.draw_line(frame, person, 5, 11) # 左肩-左臀
            self.draw_line(frame, person, 6, 12) # 右肩-右臀
            self.draw_line(frame, person, 11, 12) # 臀部
            
            # 手臂
            self.draw_line(frame, person, 5, 7)  # 左肩-左肘
            self.draw_line(frame, person, 7, 9)  # 左肘-左手腕
            self.draw_line(frame, person, 6, 8)  # 右肩-右肘
            self.draw_line(frame, person, 8, 10) # 右肘-右手腕
            
            # 腿部
            self.draw_line(frame, person, 11, 13) # 左臀-左膝
            self.draw_line(frame, person, 13, 15) # 左膝-左脚踝
            self.draw_line(frame, person, 12, 14) # 右臀-右膝
            self.draw_line(frame, person, 14, 16) # 右膝-右脚踝
            
            # 显示检测结果
            if any(violence_results.values()):
                status = "Violence Detected!"
                color = (0, 0, 255)  # 红色
                
                # 显示具体暴力类型
                y_offset = 40
                for action, detected in violence_results.items():
                    if detected:
                        cv2.putText(frame, f"{action.upper()}!", 
                                   (int(bbox[0]), int(bbox[1]) + y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                   self.colors[action], 2)
                        y_offset += 30
            else:
                status = "Normal"
                color = (0, 255, 0)  # 绿色
            
            cv2.putText(frame, status, (int(bbox[0]), int(bbox[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 显示时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (width - 300, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def draw_line(self, frame, person, idx1, idx2):
        """在两个关键点之间绘制线段"""
        kpt1 = person["keypoints"][idx1]
        kpt2 = person["keypoints"][idx2]
        
        cv2.line(frame, 
                    (int(kpt1[0]), int(kpt1[1])), 
                    (int(kpt2[0]), int(kpt2[1])), 
                    (0, 255, 255), 2)
    
    def trigger_alert(self, violence_type):
        """触发报警"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[ALERT] {timestamp} - {violence_type.upper()} detected!")
            self.last_alert_time = current_time
            return True
        return False
    
    def process_frame(self, results):
        """处理单个视频帧"""
        
        # 解析结果
        people = []
        person_res = {"bbox": [], "action": "normal"}
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                # 只考虑置信度高的检测结果
                if confidences[i] > 0.5:
                    person = {
                        "bbox": boxes[i],
                        "keypoints": keypoints[i],
                        "confidence": confidences[i]
                    }
                    people.append(person)
        
        # 检测暴力行为
        violence_results = {"throw": False, "kick": False, "drop": False}
        if len(people) > 1:
            # 只考虑主要人物（面积最大的）
            main_person = max(people, key=lambda p: (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]))
            self.history.append(main_person["keypoints"].copy())
            
            # 检测暴力行为
            violence_results = self.detect_violence(main_person["keypoints"], len(self.history))
            person_res["bbox"] = main_person["bbox"].tolist()

            for action, detected in violence_results.items():
                if detected:
                    self.trigger_alert(action)
                    person_res["action"] = action
        
        return person_res

def main():
    detector = ViolenceDetector()
    cap = cv2.VideoCapture("/home/tsingtec/wen/demo/Action_Recog/data/packagesVsorting.mp4")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('outputsorting.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, _ = detector.process_frame(frame)
        out.write(processed_frame)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()