import os
import git


git_url = 'https://github.com/ultralytics/yolov5.git'

if not os.path.exists('./Yolo_training/yoloV5'):
    git.Git('./Yolo_training').clone(git_url)