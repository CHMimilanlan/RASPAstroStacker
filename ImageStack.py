import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import shutil
import itertools
import cv2
import traceback

from utils.detect import *
from utils.starlet import *
from utils.stretch import *
from utils.transform import *


class ImageStackObject():
    """
    对于ImageStack，首先load image，先默认都是fits格式
    """

    def __init__(self, dir_path: Path, debug_tmp_path: str = "debug_tmp"):
        self.dir_path = dir_path
        self.debug_tmp_path = debug_tmp_path
        self.clean_cache()
        self.fits_path_list = self.load_fits()
        self.process_length = len(self.fits_path_list)
        self.make_cache()

        self.debayer_images_list = [None] * self.process_length
        self.stretch_images_list = [None] * self.process_length
        self.star_obj_list_list = [None] * self.process_length
        # self.stretch_grayimg_list = [None] * len(self.fits_path_list)
        self.chosen_scale_result_list = [None] * self.process_length
        self.triangle_list_list = [None] * self.process_length
        self.realtime_stack_img = None
        self.realtime_stack_img_st_debug = None
        self.rgb_flag = True

    # Todo 加入更多格式
    def load_fits(self):
        dir_fits = os.listdir(self.dir_path)
        fits_list = []
        for fits_name in dir_fits:
            fits_path = os.path.join(self.dir_path, fits_name)
            fits_path = Path(fits_path)
            fits_list.append(fits_path)
        return fits_list

    def make_cache(self):
        for fits_path in self.fits_path_list:
            fits_name = fits_path.stem
            fits_tmp_path = os.path.join(self.debug_tmp_path, fits_name)
            if not os.path.exists(fits_tmp_path):
                os.mkdir(fits_tmp_path)

    def ImageStretch(self, filepath: Path, do_jpg: bool = True, resize_size: int = 65535):
        img, continue_process, header = Debayer(filepath)
        if continue_process:
            scale = CalScale(img, resize_size)
            if scale < 1:
                resize_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                resize_img = img

            debayer_img = resize_img.copy()
            if 'BAYERPAT' in header:
                self.rgb_flag = True
                stretch_img = ComputeAndStretch_ThreeChannels(resize_img, do_jpg)
            else:
                self.rgb_flag = False
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

            return debayer_img, stretch_img
        else:
            # Todo 处理异常情况
            pass

    def ImageStretch_BatchProcess(self, do_jpg: bool = True, resize_size: int = 2048, do_debug=False):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.ImageStretch, fits_path, do_jpg, resize_size): idx
                       for idx, fits_path in enumerate(self.fits_path_list)}

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    debayer_img, stretch_img = future.result()
                    print(f"Debayer and stretch for image {idx} done, image shape-{stretch_img.shape}")
                    self.debayer_images_list[idx] = debayer_img
                    self.stretch_images_list[idx] = stretch_img
                    # self.stretch_images_list.append(result_img)
                except Exception as exc:
                    print(f'Task generated an exception: {exc}')
                    traceback.print_exc()

            executor.shutdown()

        if do_debug:
            for idx, img in enumerate(self.stretch_images_list):
                img_name = self.fits_path_list[idx].stem
                _dtp = os.path.join(self.debug_tmp_path, img_name, "stretch_img")
                self.write_img(img, _dtp)

    def clean_cache(self):
        for filename in os.listdir(self.debug_tmp_path):
            file_path = os.path.join(self.debug_tmp_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除文件夹及其所有内容
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def write_img(self, img, _dtp):
        if not os.path.exists(_dtp):
            os.mkdir(_dtp)
        cv2.imwrite(f"{_dtp}/tmp.jpg", img)

    def StarletAnalycis(self, stretch_img: np.ndarray, idx: int, reso_scale: int = None, do_debug: bool = False,
                        info_log=""):
        if len(stretch_img.shape) == 3:
            stretch_grayimg = cv2.cvtColor(stretch_img, cv2.COLOR_RGB2GRAY)
        else:
            stretch_grayimg = stretch_img

        # if do_debug:
        #     cv2.imwrite(f"{debug_tmp_path}/gray_img.jpg", stretch_grayimg)
        img_name = self.fits_path_list[idx].stem
        _dtp = os.path.join(self.debug_tmp_path, img_name)
        multiscale_results, multiscale_results_thres = starlet_transform(stretch_grayimg, reso_scale, do_debug=do_debug,
                                                                         debug_tmp_path=_dtp)
        chosen_scale_result = choose_scale_result(multiscale_results_thres, chose_type="white_pixel_count")
        star_obj_list = extract_stars(chosen_scale_result, stretch_grayimg, info_log, do_debug)
        return star_obj_list, stretch_grayimg, chosen_scale_result

    def StarletAnalycis_BatchProcess(self, reso_scale=None, do_debug: bool = False, info_log=""):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.StarletAnalycis, stretch_image, idx, reso_scale, do_debug, info_log): idx
                       for
                       idx, stretch_image in enumerate(self.stretch_images_list)}

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    star_obj_list, grayimg, thres_map = future.result()
                    print(f"Starlet Analysis for image {idx} done, star_list_num:{len(star_obj_list)}")
                    self.star_obj_list_list[idx] = star_obj_list
                    self.chosen_scale_result_list[idx] = thres_map
                except Exception as exc:
                    print(f'Task generated an exception: {exc}')
                    traceback.print_exc()

            executor.shutdown()

        print("Starlet done!")
        if do_debug:
            for idx, tmp_img in enumerate(self.chosen_scale_result_list):
                img_path = self.fits_path_list[idx]
                img_name = img_path.stem
                _dtp = os.path.join(self.debug_tmp_path, img_name, "starlet_chosen_star")
                self.write_img(tmp_img, _dtp)

    def MultiStarsRedirection(self, idx, roi_percentage: float = 0.7, top_n=10, do_debug: bool = False):
        """
        定义ROI
        # 默认使用三星校准
        校准步骤：
        1. 星点检测，提取图像中最亮的5颗星，用这五颗星中可以构成10个三角形，并加入到集合中
        2. 计算集合中所有三角形的不变量元组，对于每个三角形，计算一组不变量（invariants）。不变量是指在某些变换下保持不变的量，
        比如三角形的边长、角度、面积等。在计算机视觉中，常用的三角形不变量可能包括边长比例、角度等，这些不变量可以唯一地描述一个三角形的形状，而不受其位置、旋转和缩放的影响。
        3. 将不变量元组加入到一个KD-Tree中
        4. 删除重复的三角形元组（为什么会有重复的？？？还是说是相似的？）
        5. 从两个图片构成的不变量元组中找出所有的相似的不变量元组，并将他们匹配起来（利用kd tree加速搜索匹配）
        6. 根据匹配的不变量元组对，将一对三角形的顶点对应起来（可以根据边长对应关系，来匹配顶点）
        7. 从三个顶点的对应关系中导出一个仿射变换，并运用这个变换，同时再检查其余点，看看有多少点能够适配这个变换，这里的适配性可以通过计算变换后的点与实际点之间的距离来判断。
        如果距离足够小，则认为该点适配这个变换。（但是，在图像叠加算法中，对于星点配准的精度要求应该是很高的）如果这个变换能很好地适配其余的点（至少80%或至少10个点，以较小者为准）
        ，就接受这个变换并返回变换后的图像
        （在图像的线性平移中有一个转换公式，具体见paper）
        考虑自动识别异常图像，例如匹配点过少，那么就丢弃这张图片

        8. 在精细化的图像配准中，可以采用图像互相关算法

        :return:
        """
        # images_num = self.process_length
        # for i in range(images_num):
        # print(f"Processing StarDetect image --- {idx}")
        stretch_img = self.stretch_images_list[idx]
        star_obj_list = self.star_obj_list_list[idx]
        img_name = self.fits_path_list[idx].stem
        # 下面的star obj都经过了redirection
        roi_star_obj_list = StarObjList_ROICheck_Redirection(star_obj_list, roi_percentage, stretch_img, img_name,
                                                             do_debug, self.debug_tmp_path)
        # Todo 目前先限制了roi内星点数必须大于top_n，后续需要修改的话再在这里入手修改
        # assert len(roi_star_obj_list) >= top_n
        roi_star_obj_list.sort(key=lambda x: x["lightness"], reverse=True)
        chosen_stars_list = roi_star_obj_list[:top_n]
        self.star_obj_list_list[idx] = chosen_stars_list
        if do_debug:
            tmp_thres = self.chosen_scale_result_list[idx]
            StarDetect_DebugTmp(tmp_thres, stretch_img, chosen_stars_list, roi_star_obj_list, img_name,
                                self.debug_tmp_path)

    def MultiStarsRedirection_BatchProcess(self, roi_percentage: float = 0.7, top_n=10, do_debug: bool = False):
        length = self.process_length
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.MultiStarsRedirection, idx, roi_percentage, top_n, do_debug): idx
                       for idx in range(length)}

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                    # Todo 添加时间返回值
                    print(f"MultiStars Redirection for image {idx} done ")
                except Exception as exc:
                    print(f'Task generated an exception: {exc}')
                    traceback.print_exc()

            executor.shutdown()

        if do_debug:
            for idx, img in enumerate(self.stretch_images_list):
                img_name = self.fits_path_list[idx].stem
                _dtp = os.path.join(self.debug_tmp_path, img_name, "stretch_img")
                self.write_img(img, _dtp)

    def Triangle_Analysis(self, idx: int, do_debug: bool = False):
        # Todo 给出多种叠加区域的选择方法，既然能使用for循环递归了，那就可以实现交集、并集、参考图大小等方法，参考siril网站：https://siril.readthedocs.io/en/latest/preprocessing/registration.html，章节Apply Existing registration
        # Todo 在三角形的配准中，需要想一个算法来尽量降低O(N^2)的计算复杂度
        # Todo 考虑同一张图片中出现两个相似的三角形
        # Todo 考虑用KD-Tree优化，K=4，即一颗星与其紧邻的4个星，总共组成5颗星，能够组成10个三角形不变元组，将这几个元组加入到KD-Tree中
        # length = self.process_length
        # for i in range(length):
        invariable_tuple_list = []
        star_list_for_single_image = self.star_obj_list_list[idx]
        single_length = len(star_list_for_single_image)
        index_list = [i for i in range(single_length)]
        # 3表示选3个元素，因为三角形有三个顶点
        combinations = list(itertools.combinations(index_list, 3))
        for comb in combinations:
            point0 = star_list_for_single_image[comb[0]]
            point1 = star_list_for_single_image[comb[1]]
            point2 = star_list_for_single_image[comb[2]]
            invariant_obj = Invariant_Properties_Calculation(
                [point0["center_cor"], point1["center_cor"], point2["center_cor"]])
            invariable_tuple_list.append(invariant_obj)

        invariable_tuple_list.sort(key=lambda x: x["perimeter"])
        self.triangle_list_list[idx] = invariable_tuple_list

    def Triangle_Analysis_BatchProcess(self, do_debug):
        length = self.process_length
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.Triangle_Analysis, idx, do_debug): idx
                       for idx in range(length)}

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                    # Todo 添加时间返回值
                    print(f"Triangle_Analysis for image {idx} done ")
                except Exception as exc:
                    print(f'Task generated an exception: {exc}')
                    traceback.print_exc()

            executor.shutdown()

        if do_debug:
            for idx, img in enumerate(self.stretch_images_list):
                img_name = self.fits_path_list[idx].stem
                _dtp = os.path.join(self.debug_tmp_path, img_name, "stretch_img")
                self.write_img(img, _dtp)

    def Core_RASP(self, reference_index: int = 0, perimeter_tolerance: float = 3.0,
                  angle_tolerance: float = 1.0, side_tolerance: float = 1.0, do_debug: bool = False):
        # Todo tolerance的选择需要斟酌，应该要尽可能小
        """
        核心函数——Registration, Alignment and realtime Stacking Process(RASP)
        1. 按照三角不变元组计算所有图像相对于参考图像的配准星点
        2. 根据配准星点，计算星点的偏移量
        :param reference_index: 默认选择第0个img作为参考图
        :param perimeter_tolerance:
        :param angle_tolerance:
        :param side_tolerance:
        :param do_debug:
        :return:
        # Todo Refernce Imgae 不能直接取第0个，需要用程序自动寻找最佳的参考图片，要求参考图片位于所有图片中最居中的位置。
        """
        reference_img = self.debayer_images_list[reference_index]
        reference_img_st = self.stretch_images_list[reference_index]
        self.realtime_stack_img_st_debug = reference_img_st.copy().astype(np.uint32)
        self.realtime_stack_img = reference_img.copy().astype(np.uint32)

        reference_triangle_objs = self.triangle_list_list[reference_index]
        for i in range(self.process_length):
            print(f"=============Processing {i}-th stack=============")
            if i == reference_index:
                self.Update_Realtime_View(i, "debayer", do_debug=do_debug)
                self.Update_Realtime_View(i, "stretch","realtime_stack_debug", do_debug)
                continue
            # if i == 2:
            #     print("This is Debug index")
            compare_triangle_objs = self.triangle_list_list[i]
            compare_stretch_image = self.stretch_images_list[i]
            compare_debayer_image = self.debayer_images_list[i]
            # 获取配对的元组组合
            registration_pair_list = Triangle_Registration(reference_triangle_objs, compare_triangle_objs,
                                                           perimeter_tolerance, angle_tolerance, side_tolerance)

            cv2.imwrite("11.jpg", compare_stretch_image)

            if do_debug:
                trans_compr_img, trans_flag_map, trans_compr_img_st = Homographies_Transformation(compare_stretch_image,
                                                                                                  compare_debayer_image,
                                                                                                  registration_pair_list,
                                                                                                  do_debug,
                                                                                                  self.debug_tmp_path)
                self.realtime_stack_img_st_debug = Add_Integration(self.realtime_stack_img_st_debug, trans_compr_img_st, trans_flag_map, i)
                self.Update_Realtime_View(i, "stretch","realtime_stack_debug", do_debug)

            else:
                trans_compr_img, trans_flag_map = Homographies_Transformation(compare_stretch_image,
                                                                              compare_debayer_image,
                                                                              registration_pair_list, do_debug,
                                                                              self.debug_tmp_path)

            self.realtime_stack_img = Add_Integration(self.realtime_stack_img, trans_compr_img, trans_flag_map, i)
            self.Update_Realtime_View(i,"debayer", do_debug=do_debug)

        return self.realtime_stack_img

    def Update_Realtime_View(self, stack_round, type_code,realtime_stack_path="realtime_stack",do_debug: bool = False):
        # 调用一次该函数更新前端显示的self.realtime_stack_img，逻辑上需要这个
        if type_code=="debayer":
            stretch_realtime_stack = Stretch_RealtimeStack(self.realtime_stack_img.copy().astype(np.uint16),
                                                           self.rgb_flag)
            realtime_stack_path = os.path.join(self.debug_tmp_path, realtime_stack_path)
            if not os.path.exists(realtime_stack_path):
                os.mkdir(realtime_stack_path)

            cv2.imwrite(f"{realtime_stack_path}/{stack_round}.jpg", stretch_realtime_stack)
            print(f"Write image to {realtime_stack_path}/{stack_round}.jpg")
        elif type_code=="stretch":
            stretch_realtime_stack = self.realtime_stack_img_st_debug.astype(np.uint8)
            realtime_stack_path = os.path.join(self.debug_tmp_path, realtime_stack_path)
            if not os.path.exists(realtime_stack_path):
                os.mkdir(realtime_stack_path)
            cv2.imwrite(f"{realtime_stack_path}/{stack_round}.jpg", stretch_realtime_stack)
            print(f"Write image to {realtime_stack_path}/{stack_round}.jpg")

        else:
            assert "Error code"



    def ImageStackProcess(self, do_debug):
        self.ImageStretch_BatchProcess(do_debug=do_debug)
        self.StarletAnalycis_BatchProcess(do_debug=do_debug)
        self.MultiStarsRedirection_BatchProcess(do_debug=do_debug, top_n=10)
        self.Triangle_Analysis_BatchProcess(do_debug=do_debug)
        result_img = self.Core_RASP(do_debug=do_debug)
        stretch_realtime_stack = Stretch_RealtimeStack(result_img.copy().astype(np.uint16), self.rgb_flag)

        return stretch_realtime_stack
