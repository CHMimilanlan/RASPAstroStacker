import numpy as np
import math
import yaml
import cv2

with open(r'C:\Workman02\python\OtherProject\ImageStack\Demo01_MyStack\utils\config.yaml', 'r') as file:
    config = yaml.safe_load(file)
min_aspect_ratio = config['thresholds']['min_aspect_ratio']
min_distri_thresh = config['thresholds']['min_distri_thresh']
min_dim_ratio = config['thresholds']['min_dim_ratio']
min_circularity = config['thresholds']['min_circularity']
max_circularity = config['thresholds']['max_circularity']
mad_threshold = config['thresholds']['mad_threshold']


def CheckDistribution(stretch_img: np.ndarray):
    distri_thresh = min_distri_thresh
    max_sample = np.max(stretch_img)
    min_sample = np.min(stretch_img)
    if max_sample - min_sample < distri_thresh:
        # The distribution of pixel value is normal
        return True
    else:
        # The distribution of pixel value is too close
        return False


def Cal_Middev(mid: float, img: np.ndarray):
    mid_dev = np.median(np.abs(np.subtract(img, mid)))
    return mid_dev


def CalScale(img, resize_size: int = 2048):
    shps = img.shape
    if shps[0] > shps[1]:
        scale = resize_size / shps[0]
    else:
        scale = resize_size / shps[1]

    return scale


def checkElongated(width: int, height: int):
    minlongratio = min_aspect_ratio
    if width > height:
        ratio = width / height
    else:
        ratio = height / width

    # print("ratio:",ratio)
    if ratio > minlongratio:
        return True
    else:
        return False


def CircularityCheck(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0

    # 判断圆形度
    if min_circularity < circularity < max_circularity:  # 圆形度接近1
        return True, circularity
    else:
        return False, circularity


def PointSpreadFunctionCheck(img: np.ndarray):
    dim_log = ""
    dimratio = min_dim_ratio
    shps = img.shape
    std = np.std(img)
    dim_log += f"dim_std: {std} \n"
    if std > dimratio:
        # std += 1e-9
        # psf = create_psf_kernel(shps, sigma=std)
        # convolved_image = img*psf
        # psf_value = np.std(convolved_image)
        # # print("psf_value:",psf_value)
        # if psf_value >= 0:
        #     return False
        # else:
        #     return True
        return dim_log, False, std
    else:
        return dim_log, True, std


def HFR_Lightness_Calculation(inImage: np.ndarray, raduis: float):
    '''
    计算半通径，公式参考自 https://www.lost-infinity.com/night-sky-image-processing-part-6-measuring-the-half-flux-diameter-hfd-of-a-star-a-simple-c-implementation/
    影响hfr计算结果的因素：
        1. 星点是否处于正中心
        2. 星点提取得到的radius
    '''
    img = inImage - np.mean(inImage)
    img = np.where(img < 0, 0, img)
    shps = img.shape
    outerRadius = raduis
    # print(f"radius:{raduis}, outer:{outerRadius}")
    centerX = math.floor(shps[0] / 2.0)
    centerY = math.floor(shps[1] / 2.0)
    x = np.arange(shps[0])
    y = np.arange(shps[1])
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    distances = np.sqrt((x_grid - centerX) ** 2 + (y_grid - centerY) ** 2)
    inside = distances <= outerRadius
    sum = np.sum(img[inside])
    lightness = sum
    sumDist = np.sum(img[inside] * distances[inside])
    if sum > 0:
        hfr = sumDist / sum
    else:
        hfr = math.sqrt(2.0) * outerRadius

    # print(f"calcHfr ——  outerRadius:{outerRadius}, sumDist:{sumDist}, sum:{sum}")
    return hfr, lightness


def Add_Integration(stacking_img, trans_cmp_img, trans_flag_map, stack_round, add_type="average"):
    # trans_cmp_img = trans_cmp_img.astype(np.uint32)
    # result_img = (stacking_img * (stack_round) + trans_cmp_img) / (stack_round + 1)
    result_img = np.where(
        trans_flag_map == True,
        (stacking_img * (stack_round) + trans_cmp_img) / (stack_round + 1),
        stacking_img
    )

    return result_img
