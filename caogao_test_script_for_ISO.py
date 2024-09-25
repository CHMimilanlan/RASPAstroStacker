import cv2

from ImageStack import ImageStackObject as ISO
from pathlib import Path
import os
import shutil

def ExtractJPG(debug_tmp_path, type):
    idx = 0
    dirs = os.listdir(debug_tmp_path)
    dst_dir_path = f"{debug_tmp_path}/extract"
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)
    for dir_name in dirs:
        path_lib_dir_name = Path(dir_name)
        if path_lib_dir_name.suffix == ".jpg":
            continue

        dir_path = os.path.join(debug_tmp_path, dir_name, type)
        for file in os.listdir(dir_path):
            file_pathlib = Path(file)
            if file_pathlib.suffix == ".jpg":
                print(file)
                ori = os.path.join(dir_path, file)
                dst = os.path.join(dst_dir_path, f"{idx}.jpg")
                shutil.copy(ori, dst)
                idx += 1


fits_path = r"C:\Workman02\python\OtherProject\ImageStack\Demo01_MyStack\test_input"
dp = "debug_tmp"
do_debug = True
fits_path = Path(fits_path)

iso = ISO(fits_path, dp)
res = iso.ImageStackProcess(do_debug)
cv2.imwrite("stack_res.jpg", res)
# ExtractJPG(dp, "_star_mark")

