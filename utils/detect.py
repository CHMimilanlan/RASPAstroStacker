from .starlet import *
import time
import cv2
import math
import os
import numpy as np
from .base import *


def extract_stars(starlet_res_img, grayimg, info_log, do_debug):
    _, thres_img = cv2.threshold(starlet_res_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thres_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    stand_size = 1552
    img_shps = starlet_res_img.shape
    sclsize = np.maximum(img_shps[1], img_shps[0])
    maximun_area = 20000 * (sclsize / stand_size)
    minimun_area = 5 * math.ceil(sclsize / stand_size)
    star_obj_list = []
    star_index = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= minimun_area and area < maximun_area:
            info_log += f"==============area:{area}====================\n"
            # 计算边缘点的最小外接圆
            (x_c, y_c), circle_radius = cv2.minEnclosingCircle(contour)
            # 计算边缘点的外接矩形，这里的x y表示的左上角顶点的坐标
            x, y, width, height = cv2.boundingRect(contour)
            center = (x + int(width / 2), y + int(height / 2))
            radius = np.maximum(width, height)
            edge_expand_pixel = radius // 2
            # 为了方便后续HFR的计算，星点需要提取成正方形
            if height > width:
                # 若高大于宽，那么就需要让宽增长使得宽=高
                left = int(x + width / 2 - height / 2 - edge_expand_pixel)
                right = int(x + width / 2 + height / 2 + edge_expand_pixel)
                up = y - edge_expand_pixel
                down = y + height + edge_expand_pixel
                if down >= img_shps[0] or up <= 0 or left <= 0 or right >= img_shps[1]:
                    info_log += "edge star!\n"
                    continue
                rect_thres_expand = thres_img[up:down, left:right]
            else:
                # 若宽大于高，就要使得高增长使得高=宽
                left = x - edge_expand_pixel
                right = x + width + edge_expand_pixel
                up = int(y + height / 2 - width / 2 - edge_expand_pixel)
                down = int(y + height / 2 + width / 2 + edge_expand_pixel)
                if up < 0 or down >= img_shps[0] or left < 0 or right >= img_shps[1]:
                    info_log += "edge star!\n"
                    continue
                rect_thres_expand = thres_img[up:down, left:right]

            if checkElongated(width, height):
                info_log += "checkElongated failed\n"
                continue

            cir_flag, circularity = CircularityCheck(contour)
            if not cir_flag:
            # if False:
                info_log += "Circle check failed \n"
                continue

            d_log, dim_check, dim_std = PointSpreadFunctionCheck(grayimg[up:down, left:right])
            info_log += d_log
            if do_debug:
                star_index += 1

            hfr, lightness = HFR_Lightness_Calculation(grayimg[up:down, left:right], circle_radius)
            if hfr < 0.05:
                continue
            star_obj = {
                "center_cor": center,
                "bounding_cor": (up, down, left, right),
                "area": area,
                "radius": radius,
                "hfr": hfr,
                "dim_std": dim_std,
                "circularity": circularity,
                "lightness": lightness
            }
            star_obj_list.append(star_obj)

        else:
            info_log += f"==============area:{area}====================\n"
            info_log += "check area failed \n"
            info_log += f"area: {area}   minimun_area:{minimun_area}    maximun_area:{maximun_area} \n"

    return star_obj_list


def StarObjList_ROICheck_Redirection(star_obj_list, roi_percentage, stretch_img, img_name, do_debug, debug_tmp_path):
    # star_obj = {
    #     "center_cor": center,
    #     "bounding_cor": (up, down, left, right),
    #     "area": area,
    #     "radius": radius,
    #     "hfr": hfr,
    #     "dim_std": dim_std,
    #     "lightness": lightness
    # }
    if len(stretch_img.shape) == 3:
        stretch_grayimg = cv2.cvtColor(stretch_img, cv2.COLOR_RGB2GRAY)
    else:
        stretch_grayimg = stretch_img

    img_shapes = stretch_img.shape
    width = img_shapes[0]
    height = img_shapes[1]
    # width_start = int(width * (1 - roi_percentage) // 2)
    # width_end = int(width_start + width * roi_percentage)
    # height_start = int(height * (1 - roi_percentage) // 2)
    # height_end = int(height_start + height * roi_percentage)
    width_start = int(width * (1 - roi_percentage) // 2)
    width_end = int(width_start + width * roi_percentage)
    height_start = int(height * (1 - roi_percentage) // 2)
    height_end = int(height_start + height * roi_percentage)


    roi_star_obj_list = []
    for star_obj in star_obj_list:
        center_cor = star_obj["center_cor"]
        radius = star_obj["radius"]
        # 判断边缘
        center_width = center_cor[0]
        center_height = center_cor[1]
        expand_edge_pixel = radius

        if (
                center_width - radius - expand_edge_pixel > height_start and center_width + radius + expand_edge_pixel < height_end
                and center_height - radius - expand_edge_pixel > width_start and center_height + radius + expand_edge_pixel < width_end):
            # 星点在ROI内
            roi_star_obj_list.append(star_obj)

    s = time.time()
    for star_idx, chosen_star in enumerate(roi_star_obj_list):
        # For循环可以直接改变列表中字典变量的键值
        re_center, re_radius, re_bounding_cor = StarCenter_Redirection_Starlet(
            stretch_grayimg[chosen_star["bounding_cor"][0]:chosen_star["bounding_cor"][1],
            chosen_star["bounding_cor"][2]:chosen_star["bounding_cor"][3]], chosen_star["center_cor"],
            None, do_debug=do_debug, debug_tmp_path=os.path.join(debug_tmp_path, img_name))

        if re_radius != -1:
            # 重定向成功，重新计算星点的hfr
            re_center_gray_star = stretch_grayimg[re_bounding_cor[0]:re_bounding_cor[1],
                                  re_bounding_cor[2]:re_bounding_cor[3]]
            # print(f"before:  hfr ---- {chosen_star['hfr']}; radius ---- {chosen_star['radius']}  ")
            hfr, lightness = HFR_Lightness_Calculation(re_center_gray_star, re_radius)
            chosen_star["center_cor"] = re_center
            chosen_star["radius"] = re_radius
            chosen_star["hfr"] = hfr
            chosen_star["bounding_cor"] = re_bounding_cor

    e = time.time()

    return roi_star_obj_list


def StarDetect_DebugTmp(tmp_thres, stretch_img, chosen_stars_list,roi_star_obj_list, img_name,debug_tmp_path):
    def write_img(img, _dtp):
        if not os.path.exists(_dtp):
            os.mkdir(_dtp)
        cv2.imwrite(f"{_dtp}/tmp.jpg", img)

    if len(stretch_img.shape) == 3:
        tmp_gray = cv2.cvtColor(stretch_img, cv2.COLOR_RGB2GRAY)
    else:
        tmp_gray = stretch_img

    for star_idx, chosen_star in enumerate(chosen_stars_list):
        mark_img = cv2.circle(stretch_img, chosen_star["center_cor"], chosen_star["radius"] + 5,
                              (0, 255, 0),
                              1)
        mark_img = cv2.putText(mark_img,
                               str(round(chosen_star["hfr"], 2)) + " " + str(
                                   round(chosen_star["circularity"], 2)),
                               chosen_star["center_cor"], cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                               (0, 255, 0), 1,
                               cv2.LINE_AA)
        _dtp = os.path.join(debug_tmp_path, img_name, "star_mark")
        write_img(mark_img, _dtp)

        # single star's debug tmp
        single_star_tmp_path = os.path.join(debug_tmp_path, img_name, "recenter")
        if not os.path.exists(single_star_tmp_path):
            os.mkdir(single_star_tmp_path)
        chosen_star_thres_visualize = tmp_thres[
                                      chosen_star["bounding_cor"][0]:chosen_star["bounding_cor"][1],
                                      chosen_star["bounding_cor"][2]:chosen_star["bounding_cor"][3]]
        re_center_gray_star = tmp_gray[chosen_star["bounding_cor"][0]:chosen_star["bounding_cor"][1],
                              chosen_star["bounding_cor"][2]:chosen_star["bounding_cor"][3]]

        cv2.imwrite(f"{single_star_tmp_path}/re_center_gray_star_{star_idx}.jpg", re_center_gray_star)
        cv2.imwrite(f"{single_star_tmp_path}/re_center_thres_star_{star_idx}.jpg",
                    chosen_star_thres_visualize)

    debug_chosen_star_list = roi_star_obj_list[:5]
    for chosen_star in debug_chosen_star_list:
        _mark_img = cv2.circle(stretch_img, chosen_star["center_cor"], chosen_star["radius"] + 5,
                               (255, 0, 0),
                               1)
        _mark_img = cv2.putText(_mark_img,
                                str(round(chosen_star["hfr"], 2)) + " " + str(
                                    round(chosen_star["lightness"], 2)),
                                chosen_star["center_cor"], cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 0, 0), 1,
                                cv2.LINE_AA)
        _dtp = os.path.join(debug_tmp_path, img_name, "_star_mark")
        write_img(_mark_img, _dtp)


def StarCenter_Redirection_Starlet(gray_star: np.ndarray, center_cor_whole, reso_scale=None,
                                   do_debug: bool = False, debug_tmp_path: str = "debug_tmp"):
    def choose_scale_result(multiscale_results_thres, chose_type="hard"):
        if chose_type == "hard":
            # Todo 经过一定的样本实验得出，绝大部分情况下，只需要选取最后一张starlet解析结果即可
            chosen_num = -1
            chosen_result = multiscale_results_thres[chosen_num]
        elif chose_type == "white_pixel_count":
            pixel_num_list = []
            for idx, thres_img in enumerate(multiscale_results_thres):
                buffer_value = thres_img.flatten()
                shps = thres_img.shape
                maxSamples = 50000
                sampleBy = 1 if shps[0] * shps[1] < maxSamples else shps[0] * shps[1] / maxSamples
                sampleBy = int(sampleBy)
                # 没有normalize
                sample_value = buffer_value[::sampleBy]
                white_count = np.sum(sample_value > 127)
                pixel_num_list.append(white_count)
                # print(f"{idx} ------ white count: {white_count}")
            chosen_num = np.argmin(np.asarray(pixel_num_list))
            chosen_result = multiscale_results_thres[chosen_num]
        else:
            chosen_result = None
        return chosen_result

    def choose_contour_result(multiscale_results_thres):

        contour_nums = []
        contours_list = []
        for idx, scale in enumerate(multiscale_results_thres):
            contours, _ = cv2.findContours(scale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print(len(contours))
            contour_nums.append(len(contours))
            contours_list.append(contours)

        chosen_idx = len(contour_nums) - 1
        idx = len(contour_nums) - 1
        for n in reversed(contour_nums):
            # 从后往前遍历，找出连续为1的数列中，最后一次出现1的位置
            if n == 1:
                chosen_idx = idx
            else:
                break
            idx -= 1
        chosen_contour = contours_list[chosen_idx]
        return chosen_contour

    def extract_and_redirection(contours):
        # Todo 假如出现了多个contour，那么我需要弹出一个warning！
        """
            星点重定向，里面涉及到两个坐标系，一个是整图坐标系，一个是单星点图坐标系，
            函数中涉及到了很多坐标变量，为了分清哪个变量对应的是哪个坐标系，我在变量后加了后缀，
            whole表示整图坐标系，single表示单星点图坐标系，另外，变量名加了re前缀的，表示重定向的坐标变量
        :param contours:
        :return:
        """
        area_list = []
        for contour in contours:
            area_list.append(cv2.contourArea(contour))

        if len(area_list) == 1:
            # 唯一值就是重定向星点
            cot = contours[0]
            re_flag = True
        elif len(area_list) == 0:
            # 重定向失败
            re_flag = False
        else:
            # 出现多个重定向点，找到面积最大的那个
            area_list_np = np.asarray(area_list)
            redirection_index = np.argmax(area_list_np)
            cot = contours[redirection_index]
            re_flag = True

        if re_flag:
            re_x_single, re_y_single, re_width, re_height = cv2.boundingRect(cot)

            re_center_x_single = re_x_single + int(re_width / 2)
            re_center_y_single = re_y_single + int(re_height / 2)
            ori_width, ori_height = gray_star.shape
            re_radius = np.maximum(re_width, re_height)
            edge_expand_pixel = re_radius
            re_center_x_whole = center_cor_whole[0] - ori_width // 2 + re_center_x_single
            re_center_y_whole = center_cor_whole[1] - ori_height // 2 + re_center_y_single
            re_center_whole = (re_center_x_whole, re_center_y_whole)
            if re_height > re_width:
                # 高大于宽，使得宽对齐高
                re_left_whole = int(re_center_x_whole - re_height / 2 - edge_expand_pixel)
                re_right_whole = int(re_center_x_whole + re_height / 2 + edge_expand_pixel)
                re_up_whole = int(re_center_y_whole - re_height / 2 - edge_expand_pixel)
                re_down_whole = int(re_center_y_whole + re_height / 2 + edge_expand_pixel)
            else:
                # 宽大于高，使得高对齐宽
                re_left_whole = int(re_center_x_whole - re_width / 2 - edge_expand_pixel)
                re_right_whole = int(re_center_x_whole + re_width / 2 + edge_expand_pixel)
                re_up_whole = int(re_center_y_whole - re_width / 2 - edge_expand_pixel)
                re_down_whole = int(re_center_y_whole + re_width / 2 + edge_expand_pixel)
            re_bounding_cor_whole = (re_up_whole, re_down_whole, re_left_whole, re_right_whole)

        else:
            re_center_whole = (-1, -1)
            re_radius = -1
            re_bounding_cor_whole = (-1, -1, -1, -1)

        return re_center_whole, re_radius, re_bounding_cor_whole

    multiscale_results, multiscale_results_thres = starlet_transform(gray_star, reso_scale, process_range=[-1],
                                                                     do_debug=do_debug, debug_tmp_path=debug_tmp_path)
    chosen_contour_result = choose_contour_result(multiscale_results_thres)
    re_center_whole, re_radius, re_bounding_cor_whole = extract_and_redirection(chosen_contour_result)
    return re_center_whole, re_radius, re_bounding_cor_whole
