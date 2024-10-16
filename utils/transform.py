import numpy as np
from .base import *
import os
import cv2


def Triangle_Registration(reference_triangle_objs: list, compare_triangle_objs: list, perimeter_tolerance,
                          angle_tolerance, side_tolerance):
    # 默认选择第0个img作为参考图
    registration_pair_list = []
    success_pair = 0
    for refer_obj in reference_triangle_objs:
        for compare_obj in compare_triangle_objs:
            # 用一个误差忍受区间来减少for循环的次数
            if refer_obj["perimeter"] - perimeter_tolerance < compare_obj["perimeter"] < refer_obj[
                "perimeter"] + perimeter_tolerance:
                # 比较每个角度的数值
                refer_angle = refer_obj["angle_value"]
                refer_sides = refer_obj["sides"]
                compare_angle = compare_obj["angle_value"]
                compare_sides = compare_obj["sides"]

                angle_compare_flag = True
                sides_compare_flag = True
                for k in range(3):
                    if not (compare_angle[k] - angle_tolerance < refer_angle[k] < compare_angle[k] + angle_tolerance):
                        angle_compare_flag = False
                        break
                    if not (compare_sides[k] - side_tolerance < refer_sides[k] < compare_sides[k] + side_tolerance):
                        sides_compare_flag = False
                        break

                if angle_compare_flag and sides_compare_flag:
                    # print(f"Find registration_pair")
                    success_pair += 1
                    registration_pair = [refer_obj, compare_obj]
                    registration_pair_list.append(registration_pair)

    print(f"Find registration pair num: {success_pair}")
    return registration_pair_list


def Invariant_Properties_Calculation(points_list: list):
    """
    对于图片中任意三点组成的三角形，具备旋转、平移、缩放不变性，需要计算三角形的不变特征，包括边长和角度
    :param points_list:
    :return:
    """
    A = points_list[0]
    B = points_list[1]
    C = points_list[2]
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    AB = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    BC = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    CA = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    perimeter = AB + BC + CA

    angle_A = math.degrees(math.acos((BC ** 2 + CA ** 2 - AB ** 2) / (2 * BC * CA)))
    angle_B = math.degrees(math.acos((AB ** 2 + CA ** 2 - BC ** 2) / (2 * AB * CA)))
    angle_C = math.degrees(math.acos((AB ** 2 + BC ** 2 - CA ** 2) / (2 * AB * BC)))

    angle_obj_A = {
        "angle_value": angle_A,
        "angle": "A",
        "point_cor": A,
        "side": BC,
    }
    angle_obj_B = {
        "angle_value": angle_B,
        "angle": "B",
        "point_cor": B,
        "side": CA,
    }
    angle_obj_C = {
        "angle_value": angle_C,
        "angle": "C",
        "point_cor": C,
        "side": AB,
    }
    angle_list = [angle_obj_A, angle_obj_B, angle_obj_C]
    angle_list.sort(key=lambda x: x["angle_value"])
    # sides_list = [AB, BC, CA]
    # sides_list.sort()

    invariant_obj = {
        "angle_value": [a["angle_value"] for a in angle_list],
        "angle_order": [a["angle"] for a in angle_list],
        "sides": [a["side"] for a in angle_list],
        "points_cor": [a["point_cor"] for a in angle_list],
        "perimeter": perimeter
    }

    return invariant_obj


def transform_equation_solve(point1, point1_prime, point2, point2_prime):
    """
    point是reference， point_prime是非参考
    我自己推的公式：
    x1' - x2' = cosα·(x1-x2) - sinα·(y1-y2)
    y1' - y2' = cosα·(y1-y2) + sinα·(x1-x2)
    :param point1:
    :param point2:
    :param point3:
    :return:
    """
    try:
        (x1, y1) = point1
        (x1_prime, y1_prime) = point1_prime
        (x2, y2) = point2
        (x2_prime, y2_prime) = point2_prime
        A_prime = x1_prime - x2_prime
        A = x1 - x2
        B_prime = y1_prime - y2_prime
        B = y1 - y2
        cos_alpha = (A_prime * A + B_prime * B) / (B * B + A * A)
        # sin_alpha = (A * B_prime - A_prime * B) / (A * A - B * B)
        if cos_alpha >= 1:
            cos_alpha = 1
        elif cos_alpha <= 0:
            cos_alpha = 0
        # if sin_alpha <= 0:
        #     sin_alpha = 0
        # elif sin_alpha >= 1:
        #     sin_alpha = 1

        # alpha = (math.acos(cos_alpha) + math.asin(sin_alpha)) / 2
        alpha = math.acos(cos_alpha)
        sin_alpha = math.sin(alpha)

        t_x_1 = x1_prime + sin_alpha * y1 - cos_alpha * x1
        t_y_1 = y1_prime - sin_alpha * x1 - cos_alpha * y1

        t_x_2 = x2_prime + sin_alpha * y2 - cos_alpha * x2
        t_y_2 = y2_prime - sin_alpha * x2 - cos_alpha * y2

        t_x = (t_x_1 + t_x_2) / 2
        t_y = (t_y_1 + t_y_2) / 2

    except:
        print("11")
        print("11")

    return alpha, t_x, t_y


def Homographies_Transformation(compr_img_st: np.ndarray, compr_img_dy: np.ndarray, registration_pair_list: list,
                                do_debug: bool = False,
                                debug_tmp_path="debug_tmp"):
    def judge_alpha(cos_list, sin_list):
        min_v = 65535
        sin = 0
        cos = 0
        for cos_v in cos_list:
            if cos_v == 1:
                cos = 1
                sin = 0
                break
            for sin_v in sin_list:
                v = math.fabs(cos_v * cos_v + sin_v * sin_v - 1)
                if v < min_v:
                    min_v = v
                    sin = sin_v
                    cos = cos_v

        return cos, sin

    def transform_equation_solve_three(point1, point1_prime, point2, point2_prime, point3, point3_prime):
        """
        point是reference， point_prime是非参考
        我自己推的公式：
        x1' - x2' = cosα·(x1-x2) - sinα·(y1-y2)
        y1' - y2' = cosα·(y1-y2) + sinα·(x1-x2)
        :param point1:
        :param point2:
        :param point3:
        :return:
        """
        (x1, y1) = point1
        (x1_prime, y1_prime) = point1_prime
        (x2, y2) = point2
        (x2_prime, y2_prime) = point2_prime
        (x3, y3) = point3
        (x3_prime, y3_prime) = point3_prime

        A_prime_1 = x1_prime - x2_prime
        A_prime_2 = x1_prime - x3_prime
        A_prime_3 = x2_prime - x3_prime
        A_1 = x1 - x2
        A_2 = x1 - x3
        A_3 = x2 - x3
        B_prime_1 = y1_prime - y2_prime
        B_prime_2 = y1_prime - y3_prime
        B_prime_3 = y2_prime - y3_prime
        B_1 = y1 - y2
        B_2 = y1 - y3
        B_3 = y2 - y3

        cos_alpha_1 = (A_prime_1 * A_1 + B_prime_1 * B_1) / (B_1 * B_1 + A_1 * A_1)
        cos_alpha_2 = (A_prime_2 * A_2 + B_prime_2 * B_2) / (B_2 * B_2 + A_2 * A_2)
        cos_alpha_3 = (A_prime_3 * A_3 + B_prime_3 * B_3) / (B_3 * B_3 + A_3 * A_3)
        sin_alpha_1 = (A_1 * B_prime_1 - A_prime_1 * B_1) / (A_1 * A_1 - B_1 * B_1)
        sin_alpha_2 = (A_2 * B_prime_2 - A_prime_2 * B_2) / (A_2 * A_2 - B_2 * B_2)
        sin_alpha_3 = (A_3 * B_prime_3 - A_prime_3 * B_3) / (A_3 * A_3 - B_3 * B_3)
        cos_alpha_1 = np.clip(cos_alpha_1, -1, 1)
        cos_alpha_2 = np.clip(cos_alpha_2, -1, 1)
        cos_alpha_3 = np.clip(cos_alpha_3, -1, 1)
        sin_alpha_1 = np.clip(sin_alpha_1, -1, 1)
        sin_alpha_2 = np.clip(sin_alpha_2, -1, 1)
        sin_alpha_3 = np.clip(sin_alpha_3, -1, 1)

        cos_alpha, sin_alpha = judge_alpha([cos_alpha_1, cos_alpha_2, cos_alpha_3],
                                           [sin_alpha_1, sin_alpha_2, sin_alpha_3])
        alpha = (math.acos(cos_alpha) + math.asin(sin_alpha)) / 2
        # print(f"cos_alpha_1: {cos_alpha_1} , cos_alpha_2:{cos_alpha_2} , cos_alpha_3:{cos_alpha_3} \n"
        #       f"sin_alpha_1: {sin_alpha_1} , sin_alpha_2:{sin_alpha_2} , sin_alpha_3:{sin_alpha_3} \n"
        #       f"final cos_alpha：{cos_alpha} , sin_alpha:{sin_alpha}")


        t_x_1 = x1_prime + sin_alpha * y1 - cos_alpha * x1
        t_y_1 = y1_prime - sin_alpha * x1 - cos_alpha * y1

        t_x_2 = x2_prime + sin_alpha * y2 - cos_alpha * x2
        t_y_2 = y2_prime - sin_alpha * x2 - cos_alpha * y2

        t_x = round((t_x_1 + t_x_2) / 2)
        t_y = round((t_y_1 + t_y_2) / 2)
        print("----------------------------------------------")
        print(f"final cos_alpha: {cos_alpha:<10.4f}  sin_alpha: {sin_alpha:<10.4f}  t_x_1: {t_x_1:<10.4f}  t_x_2: {t_x_2:<10.4f}  t_y_1: {t_y_1:<10.4f}  t_y_2: {t_y_2:<10.4f}")
        print(f"final t_x: {t_x:<10.4f}  final t_y: {t_y:<10.4f}")
        # print(f"t_x_1: {t_x_1} , t_x_2:{t_x_2} t_y_1: {t_y_1} , t_y_2:{t_y_2}  final t_x：{t_x} , t_y:{t_y}")

        return alpha, t_x, t_y

    def gen_homographies_transform_matrix(alpha, tx, ty, lamda=1):
        matrix = [[lamda * math.cos(alpha), -lamda * math.sin(alpha), tx],
                  [lamda * math.sin(alpha), lamda * math.cos(alpha), ty],
                  [0, 0, 1]]
        return np.asarray(matrix)

    def apply_homographies_transform_matrix(matrix: np.ndarray, img: np.ndarray):
        shps = img.shape
        height = shps[0]
        width = shps[1]
        y_coords, x_coords = np.indices((height, width))
        ones = np.ones_like(x_coords)
        coords = np.stack([x_coords, y_coords, ones], axis=-1).reshape(-1, 3)  # 形状: (height*width, 3)
        transformed_coords = matrix @ coords.T
        x_prime = transformed_coords[0, :] / transformed_coords[2, :]  # 在不缩放的情况下，该式子的分母为1
        y_prime = transformed_coords[1, :] / transformed_coords[2, :]
        x_prime = np.clip(x_prime, 0, width - 1)
        y_prime = np.clip(y_prime, 0, height - 1)
        x_prime = np.round(x_prime).astype(int)
        y_prime = np.round(y_prime).astype(int)
        output_img = np.zeros_like(img)
        output_img[y_prime, x_prime] = img[y_coords.ravel(), x_coords.ravel()]
        flag_map = np.zeros_like(img)
        flag_map[y_prime, x_prime] = True

        return output_img, flag_map

    def zscore_Remove_Outliers(data, threshold=3):
        epsilon = 1e-10  # 一个非常小的正数，用于避免零除
        mean = np.mean(data)
        std = np.std(data)
        z_scores = [(x - mean) / (std + epsilon) for x in data]
        filtered_list = [x for i, x in enumerate(data) if abs(z_scores[i]) <= threshold]
        return filtered_list

    def IQR_Remove_Outliers(data):
        # 利用四分位距法计算得到一个阈值
        q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
        return filtered_data

    def MAD_Remove_Outliers(data, threshold=mad_threshold):
        data = np.asarray(data)
        epsilon = 1e-10  # 一个非常小的正数，用于避免零除
        median = np.median(data)
        abs_deviation = np.abs(data - median)
        mad = np.median(abs_deviation)
        modified_z_scores = 0.6745 * abs_deviation / (mad+epsilon)  # 0.6745 是 Z-score 调整因子
        flag_index = modified_z_scores < threshold
        filtered_data = data[modified_z_scores < threshold]
        return filtered_data, flag_index

    # 正式流程
    """
    坐标变换矩阵中，不考虑拍摄时焦段变化的情况，λ默认为1
    :param align_img:
    :param stack_flag_arr:
    :param registration_pair_list:
    :param compare_img:
    :return:
    """
    alpha_list = []
    tx_list = []
    ty_list = []
    for registration_pair in registration_pair_list:
        refer_obj = registration_pair[0]
        compr_obj = registration_pair[1]
        """
        x′=λ(cosα⋅x−sinα⋅y)+t_x
        y′=λ(sinα⋅x+cosα⋅y)+t_y
        """
        point1 = refer_obj["points_cor"][0]
        point1_prime = compr_obj["points_cor"][0]
        point2 = refer_obj["points_cor"][1]
        point2_prime = compr_obj["points_cor"][1]
        point3 = refer_obj["points_cor"][2]
        point3_prime = compr_obj["points_cor"][2]

        # alpha1, tx_1, ty_1 = transform_equation_solve(point1, point1_prime, point2, point2_prime)
        # alpha2, tx_2, ty_2 = transform_equation_solve(point1, point1_prime, point3, point3_prime)
        # alpha3, tx_3, ty_3 = transform_equation_solve(point2, point2_prime, point3, point3_prime)
        # print(f"alpha1: {alpha1} , alpha2:{alpha2} , alpha3:{alpha3} \n"
        #       f"tx_1: {tx_1} , tx_2:{tx_2} , tx_3:{tx_3} \n"
        #       f"ty_1: {ty_1} , ty_2:{ty_2} , ty_3:{ty_3}")
        # matrix = gen_homographies_transform_matrix((alpha1 + alpha2 + alpha3) / 3, (tx_1 + tx_2 + tx_3) / 3,
        #                                            (ty_1 + ty_2 + ty_3) / 3)
        alpha, t_x, t_y = transform_equation_solve_three(point1, point1_prime, point2, point2_prime, point3,
                                                         point3_prime)
        alpha_list.append(alpha)
        tx_list.append(t_x)
        ty_list.append(t_y)

    filtered_alpha_list, flag_index = MAD_Remove_Outliers(alpha_list)
    filtered_tx_list = np.asarray(tx_list)[flag_index]
    filtered_ty_list = np.asarray(ty_list)[flag_index]

    # filtered_tx_list, flag_index = MAD_Remove_Outliers(tx_list)
    # filtered_ty_list, flag_index = MAD_Remove_Outliers(ty_list)

    alpha = np.average(filtered_alpha_list)
    t_x = np.average(filtered_tx_list)
    t_y = np.average(filtered_ty_list)
    t_x = -t_x
    t_y = -t_y

    matrix = gen_homographies_transform_matrix(alpha, t_x, t_y)

    trans_compr_img, trans_flag_map = apply_homographies_transform_matrix(matrix, compr_img_dy)

    if do_debug:
        trans_compr_img_st, _ = apply_homographies_transform_matrix(matrix, compr_img_st)
        return trans_compr_img, trans_flag_map, trans_compr_img_st
    else:
        return trans_compr_img, trans_flag_map
