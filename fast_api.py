# -*- coding: utf-8 -*-
import os
import cv2
import algo
import json
import dotenv
import uuid
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from algo.util import calculate_iou
from concurrent.futures import ThreadPoolExecutor

# 加载环境变量
dotenv.load_dotenv()

# 环境变量配置
YOLO_MODEL = os.getenv("YOLO_MODEL")
FIGHT_MODEL = os.getenv("FIGHT_MODEL")
CIGAR_MODEL = os.getenv("CIGAR_MODEL")
FACE_MODEL = os.getenv("FACE_MODEL")
PACK_MODEL = os.getenv("PACK_MODEL")

# 响应模型定义
class DetectionResult(BaseModel):
    frame_id: int
    person_det: List[float]
    face_det: List[float] = []
    brawl: bool
    smoking: bool
    violent_sorting: bool
    intrusion_det: List[Any] = []  # 固定空列表

class VideoAnalysisRequest(BaseModel):
    video_url: Optional[str] = None  # 支持URL或文件上传
    process_rate: int = 20  # 处理帧率
    brawl_threshold: float = 0.5  # 打架检测敏感度

class VideoAnalysisResponse(BaseModel):
    job_id: str
    status: str
    results: Optional[List[DetectionResult]] = None
    message: Optional[str] = None

# FastAPI 应用
app = FastAPI(
    title="视频行为分析API",
    description="检测视频中的打架、抽烟和暴力分拣行为",
    version="1.0.0"
)

# 内存中的任务存储（生产环境应使用数据库）
analysis_tasks = {}
executor = ThreadPoolExecutor(max_workers=4)  # 并发处理限制

def process_video(video_path: str, job_id: str, process_rate: int = 20, brawl_threshold: float = 0.5) -> Dict:
    """处理视频并返回分析结果"""
    try:
        analysis_tasks[job_id]["status"] = "processing"
        print(f"🚀 开始处理任务 {job_id}，视频: {video_path}")
        
        FIGHT_ON = False
        FIGHT_ON_TIMEOUT = 5  # 打架状态保持时间(秒)
        tasks = analysis_tasks[job_id]
        
        # 初始化模型
        fdet = algo.FightDetector(FIGHT_MODEL, threshold=brawl_threshold)
        yolo = algo.YoloPoseEstimation(YOLO_MODEL)
        cigar_det = algo.CigaretteDetector(CIGAR_MODEL)
        face_det = algo.FaceDetector(FACE_MODEL)
        pack_det = algo.PackageDetector(PACK_MODEL)
        act_det = algo.ViolenceDetector()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 1 
        results_list = []
        tasks["total_frames"] = count
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 跳过帧提高处理效率
            if frame_idx % process_rate != 0:
                frame_idx += 1
                tasks["processed_frames"] = frame_idx
                continue
                
            print(f"⌛ 处理任务 {job_id}，帧 {frame_idx}/{count}")
            tasks["processed_frames"] = frame_idx
            
            person_res = yolo.infer(frame) 
            pack_res = pack_det.infer(frame)
            person_act = act_det.process_frame(person_res)
            
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
                    
                    cur_frame_res = {
                        "frame_id": frame_idx,
                        "person_det": [float(x) for x in person_box],
                        "brawl": False,
                        "smoking": False,
                        "violent_sorting": False
                    }
                    
                    # 暴力分拣检测
                    if person_act and person_act.get("bbox") == person_box:
                        for pack_box in pack_res:
                            if calculate_iou(person_box, pack_box) > 0.2:
                                cur_frame_res["violent_sorting"] = True
                                break
                    
                    # 抽烟检测
                    is_person_smoking = cigar_det.infer(person_img)
                    cur_frame_res["smoking"] = bool(is_person_smoking)
                    
                    # 人脸检测
                    face_box = face_det.infer(person_img)
                    cur_frame_res["face_det"] = [float(x) for x in face_box] if face_box else []
                    
                    results_list.append(cur_frame_res)  
                
                # 打架检测
                if boxes:
                    interaction_boxes = algo.get_interaction_box(boxes)
                    both_fighting = []
                    
                    for inter_box in interaction_boxes:
                        for conf, xyn, box, identity in zip(confs, xyn, boxes, ids):
                            center_person_x, center_person_y = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
                            if inter_box[0] <= center_person_x <= inter_box[2] and inter_box[1] <= center_person_y <= inter_box[3]:
                                is_person_fighting = fdet.detect(conf, xyn)
                                both_fighting.append(is_person_fighting)
                                
                                # 更新当前帧的打架状态
                                for r in results_list[-len(boxes):]:
                                    if r["person_det"] == box:
                                        r["brawl"] = bool(is_person_fighting)
                    
                    # 如果所有在互动区域的人都打架，触发持续打架状态
                    if both_fighting and all(both_fighting):
                        FIGHT_ON = True     

            except (TypeError, IndexError) as e:
                print(f"处理帧 {frame_idx} 时出错: {str(e)}")
                # 添加空结果避免数据处理中断
                results_list.append({
                    "frame_id": frame_idx,
                    "person_det": [],
                    "brawl": False,
                    "smoking": False,
                    "violent_sorting": False,
                    "face_det": []
                })

            # 更新打架状态超时机制
            if FIGHT_ON:
                FIGHT_ON_TIMEOUT -= 0.2

            if FIGHT_ON_TIMEOUT <= 0:
                FIGHT_ON = False
                FIGHT_ON_TIMEOUT = 5
                
            frame_idx += 1
            
        cap.release()
        
        # 转换为标准的响应模型
        validated_results = []
        for res in results_list:
            validated_results.append(DetectionResult(**res))
            
        print(f"✅ 任务 {job_id} 完成! 处理 {len(results_list)} 帧")
        return {
            "status": "completed",
            "results": validated_results,
            "processed_frames": frame_idx - 1,
            "total_frames": count
        }
        
    except Exception as e:
        print(f"❌ 任务 {job_id} 失败: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }
    finally:
        # 清理任务
        if os.path.exists(video_path):
            os.unlink(video_path)

@app.post("/analyze/video", 
          response_model=VideoAnalysisResponse,
          summary="启动视频分析任务",
          status_code=202)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video_url: Optional[str] = None,
    video_file: Optional[UploadFile] = File(None),
    process_rate: int = 20,
    brawl_threshold: float = 0.5
):
    """启动视频分析任务（异步处理）"""
    job_id = str(uuid.uuid4())
    analysis_tasks[job_id] = {
        "status": "pending",
        "params": {
            "process_rate": process_rate,
            "brawl_threshold": brawl_threshold
        }
    }
    
    # 检查输入源
    video_path = None
    if video_file:
        # 保存上传的文件到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video_file.read()
            tmp.write(content)
            video_path = tmp.name
    elif video_url:
        # 简化处理：实际项目应下载URL
        video_path = video_url
    else:
        raise HTTPException(
            status_code=400,
            detail="必须提供video_url或video_file"
        )
    
    # 启动后台任务
    def run_analysis():
        result = process_video(
            video_path=video_path,
            job_id=job_id,
            process_rate=process_rate,
            brawl_threshold=brawl_threshold
        )
        analysis_tasks[job_id].update(result)
    
    executor.submit(run_analysis)
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": f"分析任务已启动，ID: {job_id}"
    }

@app.get("/analyze/result/{job_id}",
         response_model=VideoAnalysisResponse,
         summary="获取分析结果")
async def get_analysis_result(job_id: str):
    """获取视频分析任务结果"""
    task = analysis_tasks.get(job_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"任务 {job_id} 不存在"
        )
    
    if task["status"] == "pending":
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "任务正在排队等待处理"
        }
    
    if task["status"] == "processing":
        progress = (task["processed_frames"] / task["total_frames"]) * 100
        return {
            "job_id": job_id,
            "status": "processing",
            "message": f"处理中: {progress:.1f}% 完成"
        }
    
    if task["status"] == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "message": f"处理失败: {task.get('error', '未知错误')}"
        }
    
    return {
        "job_id": job_id,
        "status": "completed",
        "results": task["results"]
    }

@app.get("/analyze/preview/{job_id}",
         summary="获取分析预览视频（动画）")
async def get_analysis_preview(job_id: str):
    """生成可视化分析结果的预览视频（示例）"""
    task = analysis_tasks.get(job_id)
    if not task or task["status"] != "completed":
        raise HTTPException(
            status_code=404,
            detail="任务未完成或不存在"
        )
    
    # 这里简化处理 - 实际应生成带有标注的视频
    # 返回一个模拟的GIF动画
    return StreamingResponse(
        open("placeholder.gif", "rb"),  # 实际项目中生成真实预览
        media_type="image/gif",
        headers={"Content-Disposition": f"attachment; filename={job_id}.gif"}
    )

@app.get("/health", summary="服务健康检查")
async def health_check():
    """服务健康状态检查"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)