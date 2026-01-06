
# 行为识别
## 1.项目简介
针对输入的视频完成人员是否打架斗殴、人员是否抽烟、快递包裹是否存在暴力分拣、人脸识别、
入侵检测(暂不支持)的视频结构化解析,返回json格式文件, 耗时较高(因为视频每帧都解析).

## 2.环境安装
```bash
conda create -n det python=3.10
conda activate det
pip install -r requirements.txt
```

## 3.配置说明
见.env

## 4.使用说明
返回的结果见output.json或者main.py的line19-line28
本地测试

```bash
python main.py --video your/video/path
```

接口调用
cv_api.py中的process_video(video_path)
调用示例
```bash
from cv_api import process_video
video_path = "path/to/your/video.mp4"
results_json = process_video(video_path)
```

## 5.结果
展示的结果在[results]下

