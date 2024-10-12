from .base import *
import numpy as np
import cv2
from pathlib import Path
from astropy.io import fits
from threading import Thread
import queue
import numexpr as ne


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



def AutoParamCompute_OneChannel(img: np.ndarray):
    # Find the median sample.
    median = np.median(img)

    max_num = np.max(img)
    norm_img = img / max_num
    m_c = median / max_num

    # Find the Median deviation: 1.4826 * median of abs(sample[i] - median).
    med_dev = Cal_Middev(median, img)
    med_dev = med_dev / max_num
    MADN = 1.4826 * med_dev

    B = 0.25
    C = -2.8
    upper_half = m_c > 0.5
    shadows = 0.0 if upper_half or MADN == 0 else min(1.0, max(0.0, (m_c + C * MADN)))
    highlights = 1.0 if not upper_half or MADN == 0 else min(1.0, max(0.0, (m_c - C * MADN)))

    if not upper_half:
        X = m_c - shadows
        M = B
    else:
        X = B
        M = highlights - m_c

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

    normalized_median = medianSample / inputRange
    MADN = 1.4826 * medDev / inputRange

    B = 0.25

    upper_half = normalized_median > 0.5

    if upper_half or MADN == 0:
        shadows = 0.0
    else:
        shadows = min(1.0, max(0.0, normalized_median + -2.8 * MADN))

    if not upper_half or MADN == 0:
        highlights = 1.0
    else:
        highlights = min(1.0, max(0.0, normalized_median - -2.8 * MADN))

    if not upper_half:
        X = normalized_median - shadows
        M = B
    else:
        X = B
        M = highlights - normalized_median

    if X == 0:
        midtones = 0.0
    elif X == M:
        midtones = 0.5
    elif X == 1:
        midtones = 1.0
    else:
        midtones = ((M - 1) * X) / ((2 * M - 1) * X - M)

    return shadows, midtones, highlights


def SinglePlaneStretch(plane: np.ndarray, native_shadows: float, native_highlights: float, bool_img: np.ndarray,
                       maxOutput: int, k1: np.float64, k2: np.float64, midtones: np.float64,
                       q: queue.Queue = None):
    """
    it is the bottleneck of our algorithm
    :param plane:
    :param native_shadows:
    :param native_highlights:
    :param bool_img:
    :param maxOutput:
    :param k1:
    :param k2:
    :param midtones:
    :param q:
    :return:
    """
    epsilon = 1e-10  # 一个非常小的正数，用于避免零除

    native_shadows = np.round(native_shadows).astype(np.uint16)
    native_highlights = np.round(native_highlights).astype(np.uint16)
    downshadow = plane < native_shadows
    uphighligh = plane > native_highlights
    other = np.logical_xor(bool_img, (np.add(downshadow, uphighligh)))

    plane[downshadow] = 0
    plane[uphighligh] = maxOutput
    # plane[other] = ((plane[other] - nativeShadows) * k1 + epsilon) / (
    #         (plane[other] - nativeShadows) * k2 - midtones + epsilon)

    variables = {
        'plane_other': plane[other],
        'nativeShadows': native_shadows,
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



def Stretch_RealtimeStack(realtime_stack_img: np.ndarray, rgb_flag: bool, do_jpg: bool = True,
                          resize_size: int = 2048):
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



