import json
import logging

import cv2
import os
import numpy as np
import glob

import torch
from skimage import io
from models.plain_seg_model import SegModel
from torchvision import transforms
try:
    from osgeo import gdal
    USE_GDAL = True
except Exception as e:
    logging.warning('No GDAL Support!')
    USE_GDAL = False

logger = logging.getLogger('log')
sh = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


class AnnotateImage:

    def __init__(self,
                 path_dir, idx2color, use_gdal=None, img_formats=None,
                 img_show_mode='origin', is_save_color=False,
                 ckpt=None
                 ):
        if ckpt is not None and os.path.exists(ckpt):
            print("Found seg model at %s, loading..." % ckpt)
            self.model = SegModel.load_from_checkpoint(ckpt).eval()

        if img_formats is None:
            img_formats = ['.jpg', '.tiff']

        self.use_gdal = use_gdal
        if use_gdal is None:
            self.use_gdal = USE_GDAL
        self.parent_dir = path_dir
        self.idx2color = idx2color

        file_list = self.get_all_file(path_dir)
        self.img_list = self.filter_img(file_list, img_formats)
        self.img_list.sort()

        # label_infos.keys = ['cur_index', 'cur_file', 'cur_folder', 'max_len', 'all_files']
        self.label_infos = self.resume_label_info(self.parent_dir)
        self.img_show_mode = img_show_mode
        assert img_show_mode in ["origin", "equalize", "clip", 'b_channel', 'g_channel', 'r_channel'], \
            'img_show_mode must be one of ["origin", "equalize", "clip", "b/g/r_channel"]'
        self.img_size_infos = {
            'ori_wh': (0, 0), 'tmp_wh': (800, 600)
        }
        self.is_save_color = is_save_color

        self.flag_is_draw_finished = True
        self.temp_draw_box = [0, 0, 0, 0]
        self.cur_XY = [0, 0]

    def get_all_file(self, dir_path):
        dir_list = glob.glob(dir_path + '/*')
        files = []
        for sub_dir in dir_list:
            if os.path.isfile(sub_dir):
                files += [sub_dir]
            elif os.path.isdir(sub_dir):
                files += self.get_all_file(sub_dir)
            else:
                raise NotImplementedError
        return files

    def filter_img(self, img_files, img_formats):
        if not isinstance(img_formats, list):
            img_formats = [img_formats]
        logger.info(f'load img format: {img_formats}')
        return_img_files = []
        for img_file in img_files:
            if os.path.splitext(img_file)[-1] in img_formats:
                return_img_files.append(img_file)
        assert len(return_img_files), 'No img, Please check the folder and the img formats'
        return return_img_files

    def resume_label_info(self, path_dir):
        label_info_path = path_dir + '/.label_infos.json'

        label_infos = {
            'cur_index': 0, 'cur_file': os.path.basename(self.img_list[0]),
            'cur_folder': os.path.dirname(self.img_list[0]),
            'max_len': len(self.img_list), 'all_files': self.img_list
        }

        if os.path.exists(label_info_path):
            label_infos_tmp = json.load(open(label_info_path, 'r'))
            if label_infos_tmp.get('max_len', 0) == label_infos['max_len']:
                label_infos = label_infos_tmp

        return label_infos

    def write_resume_file_id(self):
        path = self.parent_dir + '/resume_file_id.json'
        with open(path, 'w', encoding='utf-8') as f:
            data = {'dir_id': int(self.dir_id), 'file_ids': self.file_ids}
            json.dump(data, f)

    def read_annotations(self):
        img_file = self._get_cur_file()
        mask_file = os.path.splitext(img_file)[0] + '.png'

        w, h = self._get_img_size_infos('ori_wh')
        if os.path.exists(mask_file):
            ori_mask = io.imread(mask_file, as_gray=True)
        else:
            ori_mask = np.zeros((h, w), dtype=np.uint8)
        return ori_mask

    def save_annotations(self, ori_mask, ori_mask_list, cur_ori_mask, idx2color):
        if cur_ori_mask is not None:
            ori_mask_list_tmp = ori_mask_list + [cur_ori_mask]
        else:
            ori_mask_list_tmp = ori_mask_list
        for mask_tmp in ori_mask_list_tmp:
            for k, v in idx2color.items():
                if k == 0:
                    continue
                ori_mask[mask_tmp == k] = k
        img_file = self._get_cur_file()
        mask_file = os.path.splitext(img_file)[0] + '.png'
        io.imsave(mask_file, ori_mask)
        if self.is_save_color:
            color_mask = np.zeros((*ori_mask.shape, 3), np.uint8)
            for k, v in idx2color.items():
                color_mask[ori_mask == k] = v
            io.imsave(mask_file.replace('.png', '_color.png'), color_mask)

    def ch2next(self):
        label_info_path = self.parent_dir + '/.label_infos.json'
        cur_index = self.label_infos['cur_index'] + 1
        cur_index = np.clip(cur_index, 0, self.label_infos['max_len']-1).tolist()
        self.label_infos['cur_index'] = cur_index
        self.label_infos['cur_file'] = os.path.basename(self.img_list[cur_index])
        self.label_infos['cur_folder'] = os.path.dirname(self.img_list[cur_index])
        json.dump(self.label_infos, open(label_info_path, 'w'), indent=4)

    def ch2previous(self):
        label_info_path = self.parent_dir + '/.label_infos.json'
        cur_index = self.label_infos['cur_index'] - 1
        cur_index = np.clip(cur_index, 0, self.label_infos['max_len'] - 1).tolist()
        self.label_infos['cur_index'] = cur_index
        self.label_infos['cur_file'] = os.path.basename(self.img_list[cur_index])
        self.label_infos['cur_folder'] = os.path.dirname(self.img_list[cur_index])
        json.dump(self.label_infos, open(label_info_path, 'w'), indent=4)

    def save_label_info_now(self):
        label_info_path = self.parent_dir + '/.label_infos.json'
        json.dump(self.label_infos, open(label_info_path, 'w'), indent=4)

    # 创建回调函数
    def draw_rectangle(self, event, x, y, flags, param):
        self.cur_XY = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.flag_is_draw_finished = False
            self.temp_draw_box[0:2] = [x, y]
            self.temp_draw_box[2:] = [x, y]
        # 当鼠标左键按下并移动是绘制图形。event可以查看移动，flag查看是否按下
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            self.temp_draw_box[2:] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.flag_is_draw_finished = True
            self.temp_draw_box[2:] = [x, y]
            left, top, right, bottom = self.refine_corners(*self.temp_draw_box)
            self.ori_cur_box = self.resized2ori([left, top, right, bottom])

    def refine_corners(self, x_1, y_1, x_2, y_2):
        if x_1 <= x_2:
            if y_1 <= y_2:
                left, top, right, bottom = x_1, y_1, x_2, y_2
            elif y_1 > y_2:
                left, top, right, bottom = x_1, y_2, x_2, y_1
        elif x_1 > x_2:
            if y_1 < y_2:
                left, top, right, bottom = x_2, y_1, x_1, y_2
            elif y_1 > y_2:
                left, top, right, bottom = x_2, y_2, x_1, y_1
        left = 0 if left < 0 else left
        top = 0 if top < 0 else top
        right = self.img_size_infos['tmp_wh'][0] if right > self.img_size_infos['tmp_wh'][0] else right
        bottom = self.img_size_infos['tmp_wh'][1] if bottom > self.img_size_infos['tmp_wh'][1] else bottom
        return left, top, right, bottom

    def _get_cur_file(self):
        return self.label_infos['all_files'][self.label_infos['cur_index']]

    def _get_img_size_infos(self, key):
        return self.img_size_infos[key]

    def _set_img_size_infos(self, kv: dict):
        self.img_size_infos.update(kv)

    def read_img(self):
        img_file = self._get_cur_file()
        if self.use_gdal:
            data_set = gdal.Open(img_file)
            ori_w = data_set.RasterXSize
            ori_h = data_set.RasterYSize
            ori_band = data_set.RasterCount
            img_data = np.zeros((ori_band, ori_h, ori_w), dtype=np.uint16)
            data_set.ReadAsArray(
                0, 0, ori_w, ori_h, img_data, ori_h, ori_w,
                buf_type=gdal.GDT_UInt16, resample_alg=gdal.GRIORA_Bilinear
            )
            ori_img = img_data[:3][::-1]
            ori_img = np.transpose(ori_img, (1, 2, 0))

        else:
            ori_img = io.imread(img_file, as_gray=False)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            ori_h, ori_w, _ = ori_img.shape

        self._set_img_size_infos({'ori_wh': (ori_w, ori_h)})

        # 直方图均衡
        (b, g, r) = cv2.split(ori_img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        equalize_img = cv2.merge((bH, gH, rH))

        b_channel = cv2.merge((b, b, b))
        g_channel = cv2.merge((g, g, g))
        r_channel = cv2.merge((r, r, r))

        # 部分最小值最大值截断
        min_value, max_value = np.min(ori_img), np.max(ori_img)
        min_value = (max_value - min_value) * 0.01
        max_value = max_value - (max_value - min_value) * 0.01
        assert max_value >= min_value + 1, 'clip error: min_value >= max_value'
        clip_img = (ori_img.astype(np.float32) - min_value) / (max_value - min_value)
        clip_img = (np.clip(clip_img, a_min=0, a_max=1) * 255).astype(np.uint8)

        return {'origin': ori_img, 'equalize': equalize_img, 'clip': clip_img,
                'b_channel': b_channel, 'g_channel': g_channel, 'r_channel': r_channel
                }

    def nothing(self, x):
        pass

    def nothing(self, x):
        pass

    def weight_img_mask(self, cur_resized_img, ori_mask, ori_mask_list, cur_ori_mask, alpha, idx2color: dict, tmp_wh):
        if cur_ori_mask is not None:
            ori_mask_list_tmp = ori_mask_list + [cur_ori_mask]
        else:
            ori_mask_list_tmp = ori_mask_list
        for mask_tmp in ori_mask_list_tmp:
            for k, v in idx2color.items():
                if k == 0:
                    continue
                ori_mask[mask_tmp == k] = k

        color_mask = np.zeros((*ori_mask.shape, 3), np.uint8)
        for k, v in idx2color.items():
            color_mask[ori_mask == k] = v
        color_mask = cv2.resize(color_mask, tmp_wh, cv2.INTER_NEAREST)
        img = cv2.addWeighted(cur_resized_img, 1-alpha, color_mask, alpha, 0)
        return img

    def ch_otsu_mod(self, state, usr_data):
        if state:
            self.thres_mode = 'OTSU'

    def ch_adpmean_mod(self, state, usr_data):
        if state:
            self.thres_mode = 'ADP_Mean'
            cv2.setTrackbarPos('threshold', '', 20)

    def ch_adpgaussian_mod(self, state, usr_data):
        if state:
            self.thres_mode = 'ADP_Gaussian'
            cv2.setTrackbarPos('threshold', '', 20)

    def ch_canny_mod(self, state, usr_data):
        if state:
            self.thres_mode = 'Canny'

    def ch_model_mod(self, state, usr_data):
        if state:
            self.thres_mode = 'SegModel'
            cv2.setTrackbarPos('threshold', '', 128)

    def ch_thres_INV(self, state, usr_data):
        if state:
            self.thres_mode_INV = True
        else:
            self.thres_mode_INV = False

    def ch_box_filter(self, state, usr_data):
        if state:
            self.box_filter = True
        else:
            self.box_filter = False

    def img2origin(self, state, usr_data):
        if state:
            self.img_show_mode = 'origin'

    def img2equalize(self, state, usr_data):
        if state:
            self.img_show_mode = 'equalize'

    def img2clip(self, state, usr_data):
        if state:
            self.img_show_mode = 'clip'

    def img2b(self, state, usr_data):
        if state:
            self.img_show_mode = 'b_channel'

    def img2g(self, state, usr_data):
        if state:
            self.img_show_mode = 'g_channel'

    def img2r(self, state, usr_data):
        if state:
            self.img_show_mode = 'r_channel'


    def init_windows(self):
        cv2.namedWindow('Annotation_Window', cv2.WINDOW_NORMAL| cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow('Annotation_Window', 800, 600)
        cv2.moveWindow('Annotation_Window', 50, 50)

        # 绑定事件
        cv2.setMouseCallback('Annotation_Window', self.draw_rectangle)

        cv2.createTrackbar('threshold', '', 0, 255, self.nothing)
        cv2.setTrackbarPos('threshold', '', 20)
        cv2.createTrackbar('weighted', '', 0, 255, self.nothing)
        cv2.setTrackbarPos('weighted', '', 150)
        cv2.createTrackbar('blockSize', '', 0, 255, self.nothing)
        cv2.setTrackbarPos('blockSize', '', 100)

        self.thres_mode = 'OTSU'
        self.thres_mode_INV = False
        self.flag_is_thres_value_setted = False
        self.ori_box_mask = None
        self.cur_label_id = 1
        self.ori_cur_box = None
        self.box_filter = False
        self.ori_mask_list = []
        self.delete_roi_pressed = False
        cv2.createButton("OTSU", self.ch_otsu_mod, '', cv2.QT_RADIOBOX, 0)
        cv2.createButton("ADP_Mean", self.ch_adpmean_mod, '', cv2.QT_RADIOBOX, 0)
        cv2.createButton("ADP_Gaussian", self.ch_adpgaussian_mod, '', cv2.QT_RADIOBOX, 0)
        cv2.createButton("Canny", self.ch_canny_mod, '', cv2.QT_RADIOBOX, 0)
        if hasattr(self, 'model'):
            cv2.createButton("SegModel", self.ch_model_mod, '', cv2.QT_RADIOBOX, 1)

        cv2.createButton("INV", self.ch_thres_INV, '', cv2.QT_CHECKBOX | cv2.QT_NEW_BUTTONBAR, 1)
        cv2.createButton("BoxFilter", self.ch_box_filter, '', cv2.QT_CHECKBOX, 0)

        cv2.createButton("origin", self.img2origin, '', cv2.QT_RADIOBOX | cv2.QT_NEW_BUTTONBAR, 1)
        cv2.createButton("equalize", self.img2equalize, '', cv2.QT_RADIOBOX, 0)
        cv2.createButton("clip", self.img2clip, '', cv2.QT_RADIOBOX, 0)
        cv2.createButton("b_channel", self.img2b, '', cv2.QT_RADIOBOX, 0)
        cv2.createButton("g_channel", self.img2g, '', cv2.QT_RADIOBOX, 0)
        cv2.createButton("r_channel", self.img2r, '', cv2.QT_RADIOBOX, 0)

        cv2.createTrackbar('cur_label_id', '', 0, len(self.idx2color), self.nothing)
        cv2.setTrackbarPos('cur_label_id', '', 1)

        # cv2.putText(img, f"{self.dir_id}/{len(self.dir_list)}|{self.file_ids[self.dir_id]}/{len(self.file_list)}|Class:{CLASSES[self.cur_annotate_label]}",
        #             (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)
        info_str = f"" \
                   f"N: 下一个\t" \
                   f"P: 上一个\t" \
                   f"Z: 取消\t" \
                   f"D: 删除\t" \
                   f"Y: 确定\t" \
                   f"S: 保存\t" \
                   f"ESC: 退出"
        cv2.displayStatusBar("Annotation_Window", info_str)
        cv2.imshow('Annotation_Window', np.zeros((600, 800), dtype=np.uint8))
        cv2.waitKey(1)

    def get_threshold_mask(self, cur_ori_img, ori_cur_box, thres_mode, thres_mode_INV, cur_label_id):
        left, top, right, bottom = ori_cur_box
        box_img = cur_ori_img[top: bottom, left: right, ...].copy()
        mask_tmp = np.zeros(cur_ori_img.shape[:2], dtype=np.uint8)
        thres_mode_INV = cv2.THRESH_BINARY_INV if thres_mode_INV else cv2.THRESH_BINARY
        box_img_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
        box_img_gray = cv2.GaussianBlur(box_img_gray, (5, 5), 0)

        if 'OTSU' == thres_mode:
            if not self.flag_is_thres_value_setted:
                thresh_val, _ = cv2.threshold(box_img_gray, 0, 255, thres_mode_INV + cv2.THRESH_OTSU)
                cv2.setTrackbarPos('threshold', '', int(thresh_val))
                self.flag_is_thres_value_setted = True
            value = cv2.getTrackbarPos('threshold', '')
            value, box_mask = cv2.threshold(box_img_gray, thresh=value, maxval=255, type=thres_mode_INV)
        elif "ADP_Mean" == thres_mode:
            blockSize = cv2.getTrackbarPos('blockSize', '')
            blockSize = (blockSize // 4) * 2 + 3
            value = cv2.getTrackbarPos('threshold', '')
            value = value - 20
            box_mask = cv2.adaptiveThreshold(box_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thres_mode_INV, blockSize=blockSize, C=value)

        elif "ADP_Gaussian" == thres_mode:
            blockSize = cv2.getTrackbarPos('blockSize', '')
            blockSize = (blockSize // 4) * 2 + 3
            value = cv2.getTrackbarPos('threshold', '')
            value = value - 20
            box_mask = cv2.adaptiveThreshold(box_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thres_mode_INV, blockSize=blockSize, C=value)
        elif "Canny" == thres_mode:

            blockSize = cv2.getTrackbarPos('blockSize', '')
            value = cv2.getTrackbarPos('threshold', '')
            box_mask = cv2.Canny(box_img_gray, blockSize, value)
        elif "SegModel" == thres_mode:
            value = cv2.getTrackbarPos('threshold', '')
            value = value / 255.
            img_patch = torch.from_numpy(box_img) / 255.
            img_patch = torch.permute(img_patch, (2, 0, 1))
            img_patch = img_patch.unsqueeze(0)
            img_patch = transforms.Resize(size=(16, 256))(img_patch)
            with torch.no_grad():
                box_mask = self.model(img_patch)[0, 1, ...].cpu().numpy()
            box_mask = 255*(box_mask > value).astype(np.uint8)
            box_mask = cv2.resize(box_mask, box_img.shape[::-1][1:], cv2.INTER_NEAREST)

        box_img_area = box_img.shape[0] * box_img.shape[1]
        if self.box_filter:
            kOpen = np.ones((5, 3), np.uint8)
            box_mask_tmp = cv2.morphologyEx(box_mask, cv2.MORPH_CLOSE, kOpen, iterations=2)

            contours, _ = cv2.findContours(box_mask_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            box_mask = np.zeros_like(box_mask_tmp)
            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])
                area = cv2.contourArea(contours[i])
                if (w / h > 2 or w / h < 1/2) and area > box_img_area/5:
                    cv2.drawContours(box_mask, [contours[i]], 0, 255, -1)  ##去除小面积连通域
                    # cv2.fillPoly(img_gray, [contours[k]], 0)

        box_mask[box_mask == 255] = cur_label_id
        mask_tmp[top:bottom, left:right] = box_mask
        return mask_tmp

    def resized2ori(self, ltrb):
        width_zoom = self.img_size_infos['tmp_wh'][0] / self.img_size_infos['ori_wh'][0]
        height_zoom = self.img_size_infos['tmp_wh'][1] / self.img_size_infos['ori_wh'][1]
        left = int(ltrb[0] / width_zoom)
        top = int(ltrb[1] / height_zoom)
        right = int(ltrb[2] / width_zoom)
        bottom = int(ltrb[3] / height_zoom)
        return [left, top, right, bottom]

    def ori2resized(self, ltrb):
        width_zoom = self.img_size_infos['tmp_wh'][0] / self.img_size_infos['ori_wh'][0]
        height_zoom = self.img_size_infos['tmp_wh'][1] / self.img_size_infos['ori_wh'][1]
        left = int(ltrb[0] * width_zoom)
        top = int(ltrb[1] * height_zoom)
        right = int(ltrb[2] * width_zoom)
        bottom = int(ltrb[3] * height_zoom)
        return [left, top, right, bottom]

    def push_cur_mask(self, mask):
        if mask is not None:
            self.ori_mask_list.append(mask)

    def pop_mask(self):
        if len(self.ori_mask_list):
            self.ori_mask_list.pop()

    def delete_roi(self, roi):
        left, top, right, bottom = roi
        self.ori_mask[top:bottom, left:right] = 0
        for idx in range(len(self.ori_mask_list)):
            self.ori_mask_list[idx][top:bottom, left:right] = 0

    def run(self):
        self.init_windows()
        while True:
            img_dict = self.read_img()
            ori_mask = self.read_annotations()
            while True:
                self.ori_mask = ori_mask.copy()
                win_rect = cv2.getWindowImageRect('Annotation_Window')  # x,y,w,h
                tmp_wh = tuple(win_rect[-2:])
                self._set_img_size_infos({'tmp_wh': tmp_wh})
                cur_ori_img = img_dict[self.img_show_mode]
                self.cur_resized_img = cv2.resize(cur_ori_img, tmp_wh, cv2.INTER_LINEAR)
                self.cur_label_id = cv2.getTrackbarPos('cur_label_id', '')
                assert self.cur_label_id > 0, 'cur_label_id <= 0'

                # img = cv2.resize(img, (self.size_infos['present_width'], self.size_infos['present_height']), cv2.INTER_LINEAR)
                # tmp_bbox = self.labels.copy()
                # for bbox in tmp_bbox:
                #     width_zoom = self.size_infos['present_width'] / self.size_infos['origin_width']
                #     height_zoom = self.size_infos['present_height'] / self.size_infos['origin_height']
                #     bbox[1::2] *= width_zoom
                #     bbox[2::2] *= height_zoom
                #     class_id, left, top, right, bottom = bbox.astype(np.int)
                #     cv2.rectangle(img, (left, top), (right, bottom), SCALARS[class_id], 2)
                #     cv2.putText(img, CLASSES[class_id], (left, top), cv2.FONT_HERSHEY_COMPLEX, 0.8, SCALARS[class_id], 1)
                cur_ori_mask = None
                if self.ori_cur_box is not None:
                    box_wh = (self.ori_cur_box[2]-self.ori_cur_box[0], self.ori_cur_box[3]-self.ori_cur_box[1])
                    if self.delete_roi_pressed:
                        self.delete_roi(self.ori_cur_box)
                        self.delete_roi_pressed = False
                        self.ori_cur_box = None
                        self.save_annotations(self.ori_mask, self.ori_mask_list, cur_ori_mask, self.idx2color)
                        ori_mask = self.read_annotations()
                    else:
                        if box_wh[0] > 4 and box_wh[1] > 4:
                            cur_ori_mask = self.get_threshold_mask(cur_ori_img, self.ori_cur_box, self.thres_mode, self.thres_mode_INV, self.cur_label_id)

                alpha = cv2.getTrackbarPos('weighted', '') / 255
                img_show = self.weight_img_mask(self.cur_resized_img, self.ori_mask, self.ori_mask_list, cur_ori_mask, alpha, self.idx2color, tmp_wh)

                if not self.flag_is_draw_finished:
                    cv2.rectangle(img_show, tuple(self.temp_draw_box[:2]), tuple(self.temp_draw_box[2:]),
                                  (255, 255, 255), thickness=1)
                else:
                    if self.ori_cur_box is not None:
                        resized_cur_box = self.ori2resized(self.ori_cur_box)
                        cv2.rectangle(img_show, tuple(resized_cur_box[:2]), tuple(resized_cur_box[2:]),
                                      (255, 255, 255), thickness=1)
                cv2.line(img_show, (self.cur_XY[0], 0), (self.cur_XY[0], tmp_wh[1]), (255, 255, 255), 1)
                cv2.line(img_show, (0, self.cur_XY[1]), (tmp_wh[0], self.cur_XY[1]), (255, 255, 255), 1)
                # label_infos = {
                #             'cur_index': 0, 'cur_file': os.path.basename(self.img_list[0]),
                #             'cur_folder': os.path.dirname(self.img_list[0]),
                #             'max_len': len(self.img_list), 'all_files': self.img_list
                #         }
                info_str = f"{self.label_infos['cur_index']}/{self.label_infos['max_len']}|{os.path.splitext(self.label_infos['cur_file'])[0][-5:]}"
                # cv2.putText(img_show, info_str, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)
                cv2.displayOverlay("Annotation_Window", info_str)
                # cv2.putText(img,
                #             f"Img Mode: {img_modes[flag_img_mode]}",
                #             (800, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)

                cv2.imshow("Annotation_Window", img_show)
                key_pressed = cv2.waitKeyEx(1)
                undo_pressed = [ord('z'), ord('Z')]
                delete_pressed = [ord('d'), ord('D')]
                sure_pressed = [ord('y'), ord('Y')]
                next_pressed = [ord('n'), ord('N')]
                previous_pressed = [ord('p'), ord('P')]
                save_pressed = [ord('s'), ord('S')]

                esc_pressed = [27]
                if key_pressed in sure_pressed:
                    self.push_cur_mask(cur_ori_mask)
                    self.ori_cur_box = None
                elif key_pressed in undo_pressed:
                    self.pop_mask()
                elif key_pressed in delete_pressed:
                    self.delete_roi_pressed = True
                elif key_pressed in next_pressed:
                    self.save_annotations(self.ori_mask, self.ori_mask_list, cur_ori_mask, self.idx2color)
                    self.ch2next()
                    self.ori_cur_box = None
                    self.ori_box_mask = None
                    self.ori_mask_list = []
                    break
                elif key_pressed in save_pressed:
                    self.save_annotations(self.ori_mask, self.ori_mask_list, cur_ori_mask, self.idx2color)
                elif key_pressed in previous_pressed:
                    self.save_annotations(self.ori_mask, self.ori_mask_list, cur_ori_mask, self.idx2color)
                    self.ch2previous()
                    self.ori_cur_box = None
                    self.ori_box_mask = None
                    self.ori_mask_list = []
                    break
                elif key_pressed in esc_pressed or cv2.getWindowProperty('Annotation_Window', 0) == -1:
                    self.save_annotations(self.ori_mask, self.ori_mask_list, cur_ori_mask, self.idx2color)
                    self.save_label_info_now()
                    cv2.destroyAllWindows()
                    exit()



if __name__ == '__main__':
    # 可以递归遍历任意文件夹
    # N: Next 下一个文件
    # P: Previous下一个文件
    # Z: 取消一个已有的标注
    # D: 删除一个指定BOX标注
    # Y: 确定一个标注
    # S: 保存当前标注
    # ESC或关闭窗口键退出标注
    # Ctrl+P 打开自定义模式

    img_path_dir = r'D:\标图\20220801_资源_异常\data\cky_0729'

    idx2color = {
        # 0: (0, 0, 0),  # 背景不需要写
        1: (255, 255, 255),
    }
    idx2class = {
        0: 'background',
        1: 'foreground'
    }
    ckpt = 'models/ckpt/last.ckpt'
    annotator = AnnotateImage(
        path_dir=img_path_dir, use_gdal=False, idx2color=idx2color,
        img_formats='.jpg', is_save_color=True, ckpt=ckpt
    )
    annotator.run()


