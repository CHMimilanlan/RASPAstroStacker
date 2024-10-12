from .base import *
import cv2
import os


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
    return kernel


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
