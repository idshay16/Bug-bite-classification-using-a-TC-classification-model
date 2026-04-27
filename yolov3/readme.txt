git link to clone darknet on colab - https://github.com/pjreddie/darknet
Download labelImg tool with the link - https://pypi.org/project/labelImg/

Automatic bug-bite annotation helper:
python auto_annotate_bug_bites.py --images-dir custom_data --class-id 0 --preview-dir preview_labels

Useful flags:
--overwrite                 overwrite existing .txt labels
--fallback-center-box       if nothing is detected, write one centered box
--max-boxes 5               keep top N detections per image (supports multi-object images)

Run on your new dataset (overwrite + multi-object + preview):
python auto_annotate_bug_bites.py --images-dir ../Yolo_Bug_Data/bites --class-id 0 --overwrite --max-boxes 5 --preview-dir ../Yolo_Bug_Data/preview_labels

Manual annotation/fix tool (LabelImg launcher):
python -m pip install labelImg
python launch_labelimg_manual.py --images-dir ../Yolo_Bug_Data/bites --classes-file ../Yolo_Bug_Data/classes.txt --save-dir ../Yolo_Bug_Data/bites

If LabelImg install fails due Windows long-paths, use built-in OpenCV manual annotator:
python manual_annotate_cv2.py --images-dir ../Yolo_Bug_Data/bites --class-id 0

OpenCV manual annotator controls:
left-click drag = draw box
d = delete last box
c = clear all boxes
s = save
n = save and next image
p = save and previous image
q = save and quit
