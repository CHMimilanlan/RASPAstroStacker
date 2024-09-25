import time
import numpy as np
import math
import cv2
from pathlib import Path
from astropy.io import fits
from threading import Thread
import queue
import numexpr as ne
from concurrent.futures import ThreadPoolExecutor
import os
from sklearn.cluster import KMeans
import timeit


def checkElongated(width: int, height: int):
    minlongratio = 1.4
    if width > height:
        ratio = width / height
    else:
        ratio = height / width

    # print("ratio:",ratio)
    if ratio > minlongratio:
        return True
    else:
        return False


def CalScale(img, resize_size: int = 2048):
    shps = img.shape
    if shps[0] > shps[1]:
        scale = resize_size / shps[0]
    else:
        scale = resize_size / shps[1]

    return scale


def computeParamsOneChannel(img: np.ndarray):
    '''
    参考自 https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html section 8.5.7
    :param img:
    :return:
    '''
    shps = img.shape
    # width, height = img.shape
    buffer_value = img.flatten()
    # img尚未normalize
    # maxSamples决定了采样速度以及图像质量
    maxSamples = 50000
    sampleBy = 1 if shps[0] * shps[1] < maxSamples else shps[0] * shps[1] / maxSamples
    sampleBy = int(sampleBy)
    # 没有normalize
    sample_value = buffer_value[::sampleBy]
    medianSample = np.median(sample_value)

    medDev = np.median(np.abs(np.subtract(sample_value, medianSample)))

    # normalize
    if img.dtype == np.uint16:
        inputRange = 65535
    else:
        inputRange = 255

    normalizedMedian = medianSample / inputRange
    MADN = 1.4826 * medDev / inputRange

    B = 0.25

    upper_half = normalizedMedian > 0.5

    if upper_half or MADN == 0:
        shadows = 0.0
    else:
        shadows = min(1.0, max(0.0, normalizedMedian + -2.8 * MADN))

    if not upper_half or MADN == 0:
        highlights = 1.0
    else:
        highlights = min(1.0, max(0.0, normalizedMedian - -2.8 * MADN))

    if not upper_half:
        X = normalizedMedian - shadows
        M = B
    else:
        X = B
        M = highlights - normalizedMedian

    if X == 0:
        midtones = 0.0
    elif X == M:
        midtones = 0.5
    elif X == 1:
        midtones = 1.0
    else:
        midtones = ((M - 1) * X) / ((2 * M - 1) * X - M)

    return shadows, midtones, highlights


def SinglePlaneStretch(plane: np.ndarray, nativeShadows: float, nativeHighlights: float, bool_img: np.ndarray,
                       maxOutput: int, k1: np.float64, k2: np.float64, midtones: np.float64,
                       q: queue.Queue = None):
    """
    it is the bottleneck of our algorithm
    :param plane:
    :param nativeShadows:
    :param nativeHighlights:
    :param bool_img:
    :param maxOutput:
    :param k1:
    :param k2:
    :param midtones:
    :param q:
    :return:
    """
    epsilon = 1e-10  # 一个非常小的正数，用于避免零除

    nativeShadows = np.round(nativeShadows).astype(np.uint16)
    nativeHighlights = np.round(nativeHighlights).astype(np.uint16)
    downshadow = plane < nativeShadows
    uphighligh = plane > nativeHighlights
    other = np.logical_xor(bool_img, (np.add(downshadow, uphighligh)))

    plane[downshadow] = 0
    plane[uphighligh] = maxOutput
    # plane[other] = ((plane[other] - nativeShadows) * k1 + epsilon) / (
    #         (plane[other] - nativeShadows) * k2 - midtones + epsilon)

    variables = {
        'plane_other': plane[other],
        'nativeShadows': nativeShadows,
        'k1': k1,
        'epsilon': epsilon,
        'k2': k2,
        'midtones': midtones
    }
    expr = '((plane_other - nativeShadows) * k1 + epsilon) / ((plane_other - nativeShadows) * k2 - midtones + epsilon)'
    plane[other] = ne.evaluate(expr, local_dict=variables)

    if q is not None:
        q.put(plane)
    else:
        return plane


def stretchThreeChannels_threading(img, shadows: list, midtones: list, highlights: list,
                                   inputRange: int, do_jpg: bool):
    '''
    参考自 https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html section 8.5.6 公式[19]
    :param img:
    :param shadows:
    :param midtones:
    :param highlights:
    :param inputRange:
    :return:t
    '''
    dst_img = np.zeros(img.shape, dtype=np.uint16)
    b_plane = img[:, :, 0]
    g_plane = img[:, :, 1]
    r_plane = img[:, :, 2]
    shps = b_plane.shape

    if do_jpg:
        maxOutput = 255
    else:
        maxOutput = 65535

    maxInput = inputRange - 1 if inputRange > 1 else inputRange
    midtonesR = midtones[0]
    highlightsR = highlights[0]
    shadowsR = shadows[0]
    midtonesG = midtones[1]
    highlightsG = highlights[1]
    shadowsG = shadows[1]
    midtonesB = midtones[2]
    highlightsB = highlights[2]
    shadowsB = shadows[2]

    hsRangeFactorR = 1.0 if highlightsR == shadowsR else 1.0 / (highlightsR - shadowsR)
    hsRangeFactorG = 1.0 if highlightsG == shadowsG else 1.0 / (highlightsG - shadowsG)
    hsRangeFactorB = 1.0 if highlightsB == shadowsB else 1.0 / (highlightsB - shadowsB)

    nativeShadowsR = shadowsR * maxInput
    nativeShadowsG = shadowsG * maxInput
    nativeShadowsB = shadowsB * maxInput
    nativeHighlightsR = highlightsR * maxInput
    nativeHighlightsG = highlightsG * maxInput
    nativeHighlightsB = highlightsB * maxInput

    k1R = (midtonesR - 1) * hsRangeFactorR * maxOutput / maxInput
    k1G = (midtonesG - 1) * hsRangeFactorG * maxOutput / maxInput
    k1B = (midtonesB - 1) * hsRangeFactorB * maxOutput / maxInput
    k2R = ((2 * midtonesR) - 1) * hsRangeFactorR / maxInput
    k2G = ((2 * midtonesG) - 1) * hsRangeFactorG / maxInput
    k2B = ((2 * midtonesB) - 1) * hsRangeFactorB / maxInput

    bool_img = np.ones((shps[0], shps[1]), dtype=bool)

    q1 = queue.Queue()
    q2 = queue.Queue()
    q3 = queue.Queue()
    st1 = Thread(target=SinglePlaneStretch, args=(b_plane, nativeShadowsB, nativeHighlightsB, bool_img,
                                                  maxOutput, k1B, k2B, midtonesB, q1))
    st2 = Thread(target=SinglePlaneStretch, args=(g_plane, nativeShadowsG, nativeHighlightsG, bool_img,
                                                  maxOutput, k1G, k2G, midtonesG, q2))
    st3 = Thread(target=SinglePlaneStretch, args=(r_plane, nativeShadowsR, nativeHighlightsR, bool_img,
                                                  maxOutput, k1R, k2R, midtonesR, q3))
    st_list = [st1, st2, st3]
    for st in st_list:
        st.start()

    for st in st_list:
        st.join()

    dst_b_plane = q1.get()
    dst_g_plane = q2.get()
    dst_r_plane = q3.get()
    dst_img[:, :, 0] = dst_b_plane
    dst_img[:, :, 1] = dst_g_plane
    dst_img[:, :, 2] = dst_r_plane
    return dst_img


def ComputeAndStretch_ThreeChannels(img: np.ndarray, do_jpg: bool):
    # Todo 是否加入CLAHE
    if img.dtype == np.uint16:
        inputRange = 65535
    elif img.dtype == np.uint8:
        inputRange = 255
    else:
        inputRange = 65535

    shadowsB, midtonesB, highlightsB = computeParamsOneChannel(img[:, :, 0])
    shadowsG, midtonesG, highlightsG = computeParamsOneChannel(img[:, :, 1])
    shadowsR, midtonesR, highlightsR = computeParamsOneChannel(img[:, :, 2])
    dst = stretchThreeChannels_threading(img, [shadowsR, shadowsG, shadowsB],
                                         [midtonesR, midtonesG, midtonesB],
                                         [highlightsR, highlightsG, highlightsB], inputRange, do_jpg)
    # dst = stretchThreeChannels(img, [shadowsR, shadowsG, shadowsB],
    #                                      [mi12dtonesR, midtonesG, midtonesB],
    #                                      [highlightsR, highlightsG, highlightsB], inputRange, do_jpg)

    return dst


def ComputeAndStretch_ThreeChannels(img: np.ndarray, do_jpg: bool):
    # Todo 是否加入CLAHE
    if img.dtype == np.uint16:
        inputRange = 65535
    elif img.dtype == np.uint8:
        inputRange = 255
    else:
        inputRange = 65535

    shadowsB, midtonesB, highlightsB = computeParamsOneChannel(img[:, :, 0])
    shadowsG, midtonesG, highlightsG = computeParamsOneChannel(img[:, :, 1])
    shadowsR, midtonesR, highlightsR = computeParamsOneChannel(img[:, :, 2])
    dst = stretchThreeChannels_threading(img, [shadowsR, shadowsG, shadowsB],
                                         [midtonesR, midtonesG, midtonesB],
                                         [highlightsR, highlightsG, highlightsB], inputRange, do_jpg)
    # dst = stretchThreeChannels(img, [shadowsR, shadowsG, shadowsB],
    #                                      [mi12dtonesR, midtonesG, midtonesB],
    #                                      [highlightsR, highlightsG, highlightsB], inputRange, do_jpg)

    return dst


def Cal_Middev(mid: float, img: np.ndarray):
    mid_dev = np.median(np.abs(np.subtract(img, mid)))
    return mid_dev


def AutoParamCompute_OneChannel(img: np.ndarray):
    # Find the median sample.
    median = np.median(img)

    maxNum = np.max(img)
    norm_img = img / maxNum
    M_c = median / maxNum

    # Find the Median deviation: 1.4826 * median of abs(sample[i] - median).
    med_dev = Cal_Middev(median, img)
    med_dev = med_dev / maxNum
    MADN = 1.4826 * med_dev

    B = 0.25
    C = -2.8
    upper_half = M_c > 0.5
    shadows = 0.0 if upper_half or MADN == 0 else min(1.0, max(0.0, (M_c + C * MADN)))
    highlights = 1.0 if not upper_half or MADN == 0 else min(1.0, max(0.0, (M_c - C * MADN)))

    if not upper_half:
        X = M_c - shadows
        M = B
    else:
        X = B
        M = highlights - M_c

    if X == 0:
        midtones = 0.0
    elif X == M:
        midtones = 0.5
    elif X == 1:
        midtones = 1.0
    else:
        midtones = ((M - 1) * X) / ((2 * M - 1) * X - M)

    return norm_img, shadows, midtones, highlights


def Stretch_OneChannel(norm_img: np.ndarray, shadows: float, midtones: np.float64, highlights: float):
    '''
    参考自 https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html section 8.5.6 公式[19]
    '''
    shapes = norm_img.shape
    hsRangeFactor = 1.0 if highlights == shadows else 1.0 / (highlights - shadows)
    k1 = (midtones - 1) * hsRangeFactor
    k2 = ((2 * midtones) - 1) * hsRangeFactor

    downshadow = norm_img < shadows
    norm_img[downshadow] = 0

    uphighlight = norm_img > highlights
    norm_img[uphighlight] = 1

    if len(shapes) >= 2:
        bool_img = np.ones((shapes[0], shapes[1]), dtype=bool)

    others = np.logical_xor(bool_img, (np.add(downshadow, uphighlight)))
    # norm_img[others] = ((norm_img[others] - shadows)*k1)/((norm_img[others] - shadows)*k2 - midtones)

    epsilon = 1e-10  # 一个非常小的正数，用于避免零除
    norm_img[others] = np.divide((np.multiply(np.subtract(norm_img[others], shadows), k1) + epsilon),
                                 (np.subtract(np.multiply(np.subtract(norm_img[others], shadows), k2),
                                              midtones) + epsilon))

    return norm_img


def ComputeStretch_OneChannels(img: np.ndarray, do_jpg: bool):
    # Todo 是否加入CLAHE
    if do_jpg:
        dst_img = np.zeros(img.shape, dtype=np.uint8)
    else:
        dst_img = np.zeros(img.shape, dtype=np.uint16)

    norm_img, shadows, midtones, highlights = AutoParamCompute_OneChannel(img)

    plane0 = Stretch_OneChannel(norm_img, shadows, midtones, highlights)

    if do_jpg:
        dst_img[:, :] = np.multiply(plane0, 255)
    else:
        dst_img[:, :] = np.multiply(plane0, 65535)

    return dst_img


def CheckDistribution(stretch_img: np.ndarray):
    distri_thresh = 10
    max_sample = np.max(stretch_img)
    min_sample = np.min(stretch_img)
    if max_sample - min_sample < distri_thresh:
        # The distribution of pixel value is normal
        return True
    else:
        # The distribution of pixel value is too close
        return False


def SCNR_Average_Neutral(img: np.ndarray):
    """
    refer to https://pixinsight.com/doc/legacy/LE/21_noise_reduction/scnr/scnr.html
    :param img:
    :return:
    """
    B_plane = img[:, :, 0]
    G_plane = img[:, :, 1]
    R_plane = img[:, :, 2]
    mean_plane = (B_plane.astype(np.float32) + R_plane.astype(np.float32)) / 2
    mean_plane = mean_plane.astype(np.uint8)
    _G_plane = np.minimum(G_plane, mean_plane)
    img[:, :, 1] = _G_plane
    return img


def PointSpreadFunctionCheck(img: np.ndarray):
    dim_log = ""
    dimratio = 8
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


def creat_bspline_kernel(step_starlet):
    # 时间消耗可忽略不计
    C1 = 1. / 16.
    C2 = 4. / 16.
    C3 = 6. / 16.
    KSize = 4 * step_starlet + 1
    KS2 = KSize // 2
    kernel = np.zeros((KSize), dtype=np.float32)
    if KSize == 1:
        kernel[0] = 1.0
    else:
        kernel[0] = C1
        kernel[KSize - 1] = C1
        kernel[KS2 + step_starlet] = C2
        kernel[KS2 - step_starlet] = C2
        kernel[KS2] = C3
    return kernel


def Debayer(filepath: Path):
    """
    对 FITS 文件进行去拜耳处理（Debayering）。

    参数:
    filepath (Path): 输入的 FITS 文件路径。

    返回值:
    tuple: 包含以下三个元素的元组:
        - fits_img (np.ndarray or None): 处理后的图像，如果处理失败则为 None。
        - continue_process (bool): 是否继续后续处理的标志。
        - header (dict or None): FITS 文件头信息，如果处理失败则为 None。
    """
    continue_process = True
    fileExtension = filepath.suffix
    if fileExtension == ".fits" or fileExtension == ".fit" or fileExtension == ".FIT":
        img, header = fits.getdata(filepath, header=True)
        if 'BAYERPAT' in header:
            if header['BAYERPAT'] == "RGGB":
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_RGGB2BGR)
            elif header['BAYERPAT'] == "GBRG":
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_GBRG2BGR)
            elif header['BAYERPAT'] == "BGGR":
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_BGGR2BGR)
            elif header['BAYERPAT'] == "GRBG":
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_GRBG2BGR)
            else:
                # paa_info_logging.warning(f"WRONG BAYERPAT:{header['BAYERPAT']}")
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_RGGB2RGB)
            fits_img = img
        else:
            shapes = img.shape
            if len(shapes) == 2:
                # GrayScale
                fits_img = img
            else:
                # RGB scale but have no bayerpat, leading error image, no need to stretch
                rgb_channels = np.argmin(shapes)
                if rgb_channels == 0:
                    fits_img = img.transpose(1, 2, 0)
                elif rgb_channels == 1:
                    fits_img = img.transpose(0, 2, 1)
                else:
                    fits_img = img
                continue_process = False
    else:
        # Wrong fits format
        fits_img = None
        continue_process = False
        header = None

    return fits_img, continue_process, header


def starlet_transform(input_image, reso_scale=None, process_range=None, do_debug: bool = False,
                      debug_tmp_path="debug_tmp"):
    if reso_scale is None:
        reso_scale = int(np.ceil(np.log2(np.min(input_image.shape))) - 3)
        assert reso_scale > 0

    img_input = input_image.astype(np.float32)
    step_starlet = 1
    img_out = None
    multiscale_starlet_list = []
    multiscale_starlet_thres_list_in_process_range = []

    range_process_flag = False
    if process_range is None:
        process_range_for_starlet = [2, 3, 4, 5]
        range_process_flag = True

    for scale in range(reso_scale):
        if range_process_flag:
            if scale + 1 > process_range_for_starlet[-1]:
                continue

        kernel = creat_bspline_kernel(step_starlet)
        # s = time.time()
        img_out = cv2.sepFilter2D(img_input, cv2.CV_32F, kernelX=kernel, kernelY=kernel)
        # e = time.time()
        # print(f"scale: {scale} kernel.shape: {kernel.shape}  map shape:{input_image.shape}  ----   sepfilter2D time: {e-s}")
        multi_scale_img = img_input - img_out
        multiscale_starlet_list.append(multi_scale_img)
        img_input = img_out
        step_starlet *= 2

    multiscale_starlet_list.append(img_out)

    if process_range is None:
        process_range = [2, 3, 4, 5]
    if process_range == [-1]:
        process_range = [i for i in range(len(multiscale_starlet_list))]

    # s = time.time()
    for process_num in process_range:
        scale_img = multiscale_starlet_list[process_num]
        scale_img[scale_img < 0] = 0
        scale_img = scale_img.astype(np.uint8)
        _, thres_starlet_img = cv2.threshold(scale_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        multiscale_starlet_thres_list_in_process_range.append(thres_starlet_img)

    if do_debug:
        thres_list = []
        starlet_tmp_path = os.path.join(debug_tmp_path, "starlet_results")
        if not os.path.exists(starlet_tmp_path):
            os.mkdir(starlet_tmp_path)
        for idx, img_o in enumerate(multiscale_starlet_list):
            cv2.imwrite(f"{starlet_tmp_path}/{idx}_ori.jpg", img_o)
        for img_o_thres, idx in zip(multiscale_starlet_thres_list_in_process_range, process_range):
            cv2.imwrite(f"{starlet_tmp_path}/{idx}_thres.jpg", img_o_thres)

        for idx, ti in enumerate(thres_list):
            # 1. 统计白色像素数量
            buffer_value = ti.flatten()
            shps = ti.shape
            maxSamples = 50000
            sampleBy = 1 if shps[0] * shps[1] < maxSamples else shps[0] * shps[1] / maxSamples
            sampleBy = int(sampleBy)
            # 没有normalize
            sample_value = buffer_value[::sampleBy]
            white_count = np.sum(sample_value > 127)
            print(f"{idx} ------ white count: {white_count}")

    return multiscale_starlet_list, multiscale_starlet_thres_list_in_process_range


def choose_scale_result(multiscale_results_thres, chose_type):
    if chose_type == "hard":
        # Todo 默认选取第1个scale的图片，这个数字还有待商榷
        chosen_num = 1
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


# @timeit_decorator
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
                center_width - radius - expand_edge_pixel > width_start and center_width + radius + expand_edge_pixel < width_end
                and center_height - radius - expand_edge_pixel > height_start and center_height + radius + expand_edge_pixel < height_end):
            # 星点在ROI内
            roi_star_obj_list.append(star_obj)

    s = time.time()
    # for star_idx, chosen_star in enumerate(roi_star_obj_list):
    #     # For循环可以直接改变列表中字典变量的键值
    #     re_center, re_radius, re_bounding_cor = StarCenter_Redirection_Starlet(
    #         stretch_grayimg[chosen_star["bounding_cor"][0]:chosen_star["bounding_cor"][1],
    #         chosen_star["bounding_cor"][2]:chosen_star["bounding_cor"][3]], chosen_star["center_cor"],
    #         None, do_debug=do_debug, debug_tmp_path=os.path.join(debug_tmp_path, img_name))
    #
    #     if re_radius != -1:
    #         # 重定向成功，重新计算星点的hfr
    #         re_center_gray_star = stretch_grayimg[re_bounding_cor[0]:re_bounding_cor[1],
    #                               re_bounding_cor[2]:re_bounding_cor[3]]
    #         # print(f"before:  hfr ---- {chosen_star['hfr']}; radius ---- {chosen_star['radius']}  ")
    #         hfr, lightness = HFR_Lightness_Calculation(re_center_gray_star, re_radius)
    #         chosen_star["center_cor"] = re_center
    #         chosen_star["radius"] = re_radius
    #         chosen_star["hfr"] = hfr
    #         chosen_star["bounding_cor"] = re_bounding_cor

    e = time.time()
    print(f"redireciton and debug --- {e - s}")

    return roi_star_obj_list


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


# def Triangle_Registration(reference_index: int = 0, perimeter_tolerance: float = 3.0,
#                           angle_tolerance: float = 1.0, side_tolerance: float = 1.0):
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


# def Find_Reference_Image(tri_list_list,type:str):
#     if type == "hard":
#         reference_idx = 0
#     elif type == "distance":
#         for idx, tri_list in enumerate(tri_list_list):
#
#     return reference_idx

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
        print(f"final cos_alpha：{cos_alpha} , sin_alpha:{sin_alpha} \n")

        t_x_1 = x1_prime + sin_alpha * y1 - cos_alpha * x1
        t_y_1 = y1_prime - sin_alpha * x1 - cos_alpha * y1

        t_x_2 = x2_prime + sin_alpha * y2 - cos_alpha * x2
        t_y_2 = y2_prime - sin_alpha * x2 - cos_alpha * y2

        t_x = round((t_x_1 + t_x_2) / 2)
        t_y = round((t_y_1 + t_y_2) / 2)
        print(f"t_x_1: {t_x_1} , t_x_2:{t_x_2} t_y_1: {t_y_1} , t_y_2:{t_y_2}  final t_x：{t_x} , t_y:{t_y}")

        if t_x > 1000:
            print("啊？？？？？？？？？？")

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
        return output_img

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

    alpha = np.average(alpha_list)
    t_x = np.average(tx_list)
    t_y = np.average(ty_list)
    t_x = -t_x
    t_y = -t_y

    matrix = gen_homographies_transform_matrix(alpha, t_x, t_y)

    trans_compr_img = apply_homographies_transform_matrix(matrix, compr_img_st)
    if do_debug:
        print(matrix)
        align_dir = "align_dir"
        d_path = os.path.join(debug_tmp_path, align_dir)
        if not os.path.exists(d_path):
            os.mkdir(d_path)
        cv2.imwrite(f"{d_path}/res.jpg", trans_compr_img)

    return trans_compr_img


def Add_Integration(stacking_img, trans_cmp_img, stack_round, add_type="average"):
    trans_cmp_img = trans_cmp_img.astype(np.uint32)
    result_img = (stacking_img * (stack_round) + trans_cmp_img) / (stack_round + 1)
    return result_img


def Stretch_RealtimeStack(realtime_stack_img: np.ndarray, rgb_flag: bool, do_jpg: bool = True,
                          resize_size: int = 65535):
    continue_process = True
    if continue_process:
        scale = CalScale(realtime_stack_img, resize_size)
        if scale < 1:
            resize_img = cv2.resize(realtime_stack_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            resize_img = realtime_stack_img

        if rgb_flag:
            stretch_img = ComputeAndStretch_ThreeChannels(resize_img, do_jpg)
        else:
            stretch_img = ComputeStretch_OneChannels(resize_img, do_jpg)

        if do_jpg:
            if CheckDistribution(stretch_img):
                norm_img = stretch_img
            else:
                norm_img = cv2.normalize(stretch_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            norm_img = norm_img.astype(np.uint8)
            stretch_img = norm_img

        shps = stretch_img.shape
        if len(shps) == 3:
            stretch_img = SCNR_Average_Neutral(stretch_img)

        return stretch_img
    else:
        # Todo 处理异常情况
        pass
