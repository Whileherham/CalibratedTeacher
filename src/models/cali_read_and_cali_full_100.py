import pdb
from pathlib import Path

import mmcv
import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models import DETECTORS, build_detector
from torch.nn import functional as F
from .whh_utils import load_from_8pkls

from src.utils import log_every_n
from src.utils.structure_utils import dict_split
from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid
import copy

DEBUG = False  # 默认
IGNORE = True
if IGNORE:
    CLASS_ADD = 100


ifcalibrate = True
from netcal.scaling import LogisticCalibration, TemperatureScaling
from .whh_utils import match_per_iter, fit_calibrator, calibrate, get_sparse_per_iter

# DEBUG = True


# 专门为单阶段设计，没有proposal那些


@DETECTORS.register_module()
class Cali_read_full_100(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(Cali_read_full_100, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.current_iter = 0
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
            self.ori_thr = self.train_cfg.ori_thr #0.4
            self.thr_after_cali = self.train_cfg.strict_thr #0.7
            self.cali_method = self.train_cfg.cali_method
            self.cali_input = self.train_cfg.cali_input

        self.save_dict = {}


        # -----------------------------------for calibration---------------------------------------
        if ifcalibrate:
            # self.ori_thr = self.train_cfg.pseudo_label_initial_score_thr
            # self.thr_after_cali = self.train_cfg.pseudo_label_initial_score_thr  # 暂定不变，之后统一设定
            # self.thr_after_cali = 0.7  # 暂定不变，之后统一设定


            self.train_cfg.pseudo_label_initial_score_thr = self.ori_thr

            # 假设就是单输入
            self.range_iters = 500 # 拟合,每500iter拟合一次校正器
            assert self.train_cfg.save_results == True # 因为我们要读取
            self.fit_iters = 200 #假设每1iter拟合一次，但是每次都要校正

            self.start_fit_iter = 501

            if self.cali_input == 'c':
                self.n_bins = [20]
            elif self.cali_input == 'cxy' or self.cali_input == 'cwh':
                self.n_bins = [8,8,8]
                print('No not finished')
            elif self.cali_input == 'cxywh':
                self.n_bins = [5,5,5,5,5]
                print('No not finished')

            if self.cali_method == 'logistic':
                self.calibrator = LogisticCalibration(detection=True, use_cuda=True)
            elif self.cali_method == 'temp':
                self.calibrator = TemperatureScaling(detection=True, use_cuda=True)
                assert self.cali_input == 'c'
            # 下面这些用于拟合
            self.data2fit = {}
            self.finished_fit = False
            self.ori_do_merge = self.train_cfg.do_merge

            # for i in range(self.range_iters):
            #     self.data2fit[i+1] = {}
            #     self.data2fit[i+1]['matches'] = []
            #     self.data2fit[i+1]['confidences'] = []
            # x = torch.nn.BCEWithLogitsLoss(reduction='mean')(x,y)





    def forward_train(self, img, img_metas, **kwargs):
        # WHH
        # img: torch.Size([2, 3, 896, 1344]), 这里的2 是俩一样的图片（config bsz=1）但是被不同增强了
        # img_metas[0].keys() = dict_keys(['filename', 'ori_shape', 'img_shape', 'img_norm_cfg', 'pad_shape', 'scale_factor', 'tag', 'transform_matrix'])


        super().forward_train(img, img_metas, **kwargs)
        self.current_iter += 1

        # if self.current_iter<500: #------debug
        #     self.current_iter = 500
        #     self.train_cfg.save_results = False


        if self.current_iter <= 500:
            self.train_cfg.pseudo_label_initial_score_thr = self.ori_thr #低阈值 0.4
            self.train_cfg.do_merge = False # 这个时候，不进行融合！避免低阈值带崩节奏
        else:
            self.train_cfg.pseudo_label_initial_score_thr = self.ori_thr
            self.train_cfg.do_merge = self.ori_do_merge


        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        # WHH kwargs.keys() = dict_keys(['gt_bboxes', 'gt_labels', 'img', 'img_metas', 'tag'])
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        # WHH data_groups 按照weak和strong对kwargs区分一下
        loss = {}

        if self.current_iter >= self.train_cfg.mining_warmup:  # self.train_cfg.mining_warmup=0???
            # pdb.set_trace()
            if ifcalibrate:
                pseudo_bboxes, pseudo_labels, cali_bboxes = self.forward_teacher(data_groups["weak"], data_groups["strong"])
            else:
                pseudo_bboxes, pseudo_labels = self.forward_teacher(data_groups["weak"], data_groups["strong"])


            # pdb.set_trace()
            # pseudo_bboxes: list, len=1(应该是bsz) 每个都是torch.Size([0, 5])
            # pseudo_labels: tuple len=1 每个都是torch.Size([0])
            gt_bboxes, gt_labels = data_groups["strong"]["gt_bboxes"], data_groups["strong"]['gt_labels']
            # gt_bboxes:list, len=bsz per gpu torch.Size([3, 4]),3是bbox数量
            # gt_labels:len=1,torch.Size([3])
            # print('at iteration', self.current_iter)
            # print('pb,gtb=',pseudo_bboxes[0].shape,gt_bboxes[0].shape)

            log_every_n({"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)})
            # pdb.set_trace()
            if ifcalibrate:

                merge_bboxes, merge_labels = self.merge_boxes(gt_bboxes, gt_labels, pseudo_bboxes, pseudo_labels, dt_cali=cali_bboxes)

            else:
                merge_bboxes, merge_labels = self.merge_boxes(gt_bboxes, gt_labels, pseudo_bboxes, pseudo_labels)
            # merge_bboxes list len=1, torch.Size([3, 5]) 注意一下5是什么
            # merge_labels list len=1, torch.Size([3])

            # 展示目标框
            from mmcv.runner import get_dist_info
            if DEBUG and get_dist_info()[0] == 0:
                # import pdb
                # pdb.set_trace()
                for i, (mb, gb, ml, gl) in enumerate(zip(
                        merge_bboxes, gt_bboxes, merge_labels, gt_labels)):
                    if len(mb) > len(gb):
                        import numpy as np
                        from mmdet.core.visualization import imshow_det_bboxes
                        mean = np.array([[[103.53, 116.28, 123.675]]])
                        imshow_det_bboxes(
                            img=data_groups["weak"]['img'][i].cpu().numpy().transpose([1, 2, 0]) + mean,
                            bboxes=data_groups["weak"]['gt_bboxes'][i].cpu().numpy(),
                            labels=data_groups["weak"]['gt_labels'][i].cpu().numpy(),
                            class_names=self.CLASSES
                        )
                        imshow_det_bboxes(
                            img=data_groups["strong"]['img'][i].cpu().numpy().transpose([1, 2, 0]) + mean,
                            bboxes=mb[:len(gb)].cpu().numpy(),
                            labels=ml[:len(gb)].cpu().numpy(),
                            class_names=self.CLASSES,
                            bbox_color='blue'
                        )
                        imshow_det_bboxes(
                            img=data_groups["strong"]['img'][i].cpu().numpy().transpose([1, 2, 0]) + mean,
                            bboxes=mb[len(gb):].cpu().numpy(),
                            labels=ml[len(gb):].cpu().numpy(),
                            class_names=self.CLASSES,
                            bbox_color='red'
                        )

            # 用合并后的框换稀疏 GT
            if self.train_cfg.do_merge:
                data_groups["strong"]["gt_bboxes"], data_groups["strong"]['gt_labels'] = [b[:, :4] for b in
                                                                                          merge_bboxes], merge_labels


        losses = self.student.forward_train(**data_groups["strong"])
        loss.update(**losses)


        # load and fit
        if ifcalibrate:
            # ----------------------------------- calibration start2 ---------------------------------------
            # if img.device == torch.device(type='cuda', index=0): #只在0卡上做校正
            if True:
                if self.current_iter % self.range_iters == 0:
                    print('we should fit now')
                    # 1. 读取pkl
                    # 2. 过滤
                    # 3. 匹配
                    # 4. 校正
                    path_pkl = self.train_cfg.save_dir
                    if path_pkl[-1] != '/':
                        path_pkl = path_pkl + '/'
                    import os
                    for root, ds, files in os.walk(path_pkl):
                        L = len(files)
                        # assert L % 8 ==0
                        files.sort(reverse=False)
                        l = self.current_iter // self.range_iters -1 # 500iter 0
                        index_first = l * 8
                        index_last = index_first + 8
                        pkl_files_8 = files[index_first:index_last] # 8个文件
                        # print(pkl_files_8)

                    matches_8_sparse, pseudo_matches_8_left, \
                    confidences_8_sparse, confidences_8_left = \
                        load_from_8pkls(pkl_files_8, saved_path=path_pkl, tp_iou_thre_sparse=0.75)
                    # 以上四个，都是torch.Size([num])或者torch.Size([num2])
                    print('num of sparse and left are',len(confidences_8_sparse),len(confidences_8_left))
                    if len(matches_8_sparse) > 0:
                        if 0 in matches_8_sparse and 1 in matches_8_sparse:
                            self.finished_fit = True
                            input_train = confidences_8_sparse.numpy()
                            matches_train = matches_8_sparse.numpy()
                            # input_test_before = confidences_8_left.numpy()
                            # pseudo_matches_test_before = pseudo_matches_8_left.numpy()
                            # input_test_after = None
                            # pseudo_matches_test_after = pseudo_matches_test_before
                            self.calibrator.fit(input_train, matches_train)
                            print('successfully fit')
                        else:
                            print('0 or 1 are not included')
                    else:
                        print('nothing to fit')


            # ----------------------------------- calibration end2 ---------------------------------------



        return loss

    def merge_boxes(self, gt_bboxes, gt_labels, dt_boxes, dt_labels, dt_cali = None):

        # 前三个是list，后面是tuple，len=bsz=1
        # torch.Size([1, 4])  torch.Size([1])  torch.Size([0, 5]) torch.Size([0])
        # 根据下面的代码，dt bbox的格式为tlx,tly,brx,bry,左上角坐标原点

        if dt_cali == None or self.current_iter <= self.start_fit_iter or not self.finished_fit:
            new_gt_bboxes, new_gt_labels = [], []
            for gt_boxes_per_image, gt_labels_per_image, dt_boxes_per_image, dt_labels_per_image in zip(
                    gt_bboxes, gt_labels, dt_boxes, dt_labels):
                # gt_boxes_per_image torch.Size([num, 4])
                # gt_labels_per_image torch.Size([num])
                # dt_boxes_per_image torch.Size([num2, 5])
                # dt_labels_per_image torch.Size([num2])
                # print('yes')
                # pdb.set_trace()

                if self.train_cfg.mining_min_size >= 0:  # 最小挖掘的宽高，10，这个应该可以调节
                    # print(dt_boxes_per_image.shape)
                    if dt_boxes_per_image.shape[1] != 5:
                        pdb.set_trace()
                    w = dt_boxes_per_image[:, 2] - dt_boxes_per_image[:, 0]
                    h = dt_boxes_per_image[:, 3] - dt_boxes_per_image[:, 1]
                    valid_mask = (w > self.train_cfg.mining_min_size) & (h > self.train_cfg.mining_min_size)
                    if not valid_mask.all():
                        dt_boxes_per_image = dt_boxes_per_image[valid_mask]
                        dt_labels_per_image = dt_labels_per_image[valid_mask]

                target_class_list = gt_labels_per_image.reshape(-1, 1)
                pred_class_list = dt_labels_per_image.reshape(1, -1)
                class_filter = target_class_list == pred_class_list

                iob_matrix = bbox_overlaps(dt_boxes_per_image[:, :4], gt_boxes_per_image, mode='iof').T  # [gt,dt]
                iob_filter = (iob_matrix > 0.9) & class_filter  # whh:box重合且类别对得上，认为是一个 #[gt,dt]

                iof_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4], mode='iof')  # [gt,dt]
                iof_filter = (iof_matrix > 0.9) & class_filter

                iou_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4])
                iou_filter = (iou_matrix > self.train_cfg.mining_th_score) & class_filter

                final_filter = iou_filter | (iou_matrix > 0.75) | iof_filter | iob_filter  # gt dt
                # whh:这块应该是把dt和gt 大幅度相交的直接过滤掉，但是为什么要分别算俩iof，是因为只用iou不够嘛

                unlabel_idxs = torch.sum(final_filter, 0) == 0  # whh: TRUE表示剩下的bbox的ids len=dt

                if len(gt_boxes_per_image) == 0:  # whh:太惨了，应该暂时不至于
                    gt_boxes_per_image = torch.empty((0, dt_boxes_per_image.shape[1]),
                                                     dtype=dt_boxes_per_image.dtype,
                                                     layout=dt_boxes_per_image.layout,
                                                     device=dt_boxes_per_image.device)

                pad = dt_boxes_per_image.shape[1] - gt_boxes_per_image.shape[1]
                # from mmcv.runner import get_dist_info
                # if get_dist_info()[0] == 0:
                #     import pdb
                #     pdb.set_trace()
                if pad > 0:
                    # gt_boxes_per_image: torch.Size([4, 4])
                    gt_boxes_per_image = F.pad(gt_boxes_per_image, (0, pad, 0, 0), value=0)  # 变成4 5 ，最后一维度是0
                    gt_boxes_per_image[:, 4] = 1  # score to be 1, std to be 0
                # pdb.set_trace()
                if gt_boxes_per_image.shape[1] != dt_boxes_per_image[unlabel_idxs].shape[1] or len(
                        gt_labels_per_image.shape) != len(dt_labels_per_image[unlabel_idxs].shape):
                    pdb.set_trace()


                new_gt_bboxes.append(torch.cat([gt_boxes_per_image,
                                                dt_boxes_per_image[unlabel_idxs]]))

                if IGNORE:
                    good_dt = dt_labels_per_image[unlabel_idxs]
                    good_dt = good_dt + CLASS_ADD
                    new_gt_labels.append(torch.cat([gt_labels_per_image,
                                                    good_dt]))

                else:
                    new_gt_labels.append(torch.cat([gt_labels_per_image,
                                                    dt_labels_per_image[unlabel_idxs]]))


                if DEBUG and torch.sum(unlabel_idxs) > 0:
                    print("Add", sum(unlabel_idxs.int()), dt_labels_per_image[unlabel_idxs])
                    print("IoF:", ((iof_matrix * class_filter).T)[unlabel_idxs].max(1)[0])
                    print("IoB:", ((iob_matrix * class_filter).T)[unlabel_idxs].max(1)[0])
                    print("IoU:", ((iou_matrix * class_filter).T)[unlabel_idxs].max(1)[0])
                    print("Scores:", dt_boxes_per_image[unlabel_idxs][:, 4])

            return new_gt_bboxes, new_gt_labels

        else:

            # WHH update in 0217, calibration
            new_gt_bboxes, new_gt_labels = [], []
            for gt_boxes_per_image, gt_labels_per_image, dt_boxes_per_image, dt_labels_per_image, dt_cali_per_image in zip(
                    gt_bboxes, gt_labels, dt_boxes, dt_labels, dt_cali):
                # gt_boxes_per_image torch.Size([num, 4])
                # gt_labels_per_image torch.Size([num])
                # dt_boxes_per_image torch.Size([num2, 5])
                # dt_labels_per_image torch.Size([num2])
                # print('yes')
                # pdb.set_trace()



                if self.train_cfg.mining_min_size >= 0:  # 最小挖掘的宽高，10，这个应该可以调节
                    # print(dt_boxes_per_image.shape)
                    if dt_boxes_per_image.shape[1] != 5:
                        pdb.set_trace()
                    w = dt_boxes_per_image[:, 2] - dt_boxes_per_image[:, 0]
                    h = dt_boxes_per_image[:, 3] - dt_boxes_per_image[:, 1]
                    valid_mask = (w > self.train_cfg.mining_min_size) & (h > self.train_cfg.mining_min_size)
                    if not valid_mask.all():
                        dt_boxes_per_image = dt_boxes_per_image[valid_mask]
                        dt_labels_per_image = dt_labels_per_image[valid_mask]
                        dt_cali_per_image = dt_cali_per_image[valid_mask]

                target_class_list = gt_labels_per_image.reshape(-1, 1)
                pred_class_list = dt_labels_per_image.reshape(1, -1)
                class_filter = target_class_list == pred_class_list

                iob_matrix = bbox_overlaps(dt_boxes_per_image[:, :4], gt_boxes_per_image, mode='iof').T  # [gt,dt]
                iob_filter = (iob_matrix > 0.9) & class_filter  # whh:box重合且类别对得上，认为是一个 #[gt,dt]

                iof_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4], mode='iof')  # [gt,dt]
                iof_filter = (iof_matrix > 0.9) & class_filter

                iou_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4])
                iou_filter = (iou_matrix > self.train_cfg.mining_th_score) & class_filter

                final_filter = iou_filter | (iou_matrix > 0.75) | iof_filter | iob_filter  # gt dt
                # whh:这块应该是把dt和gt 大幅度相交的直接过滤掉，但是为什么要分别算俩iof，是因为只用iou不够嘛

                unlabel_idxs = torch.sum(final_filter, 0) == 0  # whh: TRUE表示剩下的bbox的ids len=dt


                # pdb.set_trace()

                confidence_cali = dt_cali_per_image[:,-1]
                # whh add把校正后的低置信度再过滤一遍
                confident_idxs = confidence_cali >= self.thr_after_cali
                if confident_idxs.shape != unlabel_idxs.shape:
                    print(confident_idxs.shape, unlabel_idxs.shape)
                    pdb.set_trace()
                unlabel_idxs = unlabel_idxs & confident_idxs



                if len(gt_boxes_per_image) == 0:  # whh:太惨了，应该暂时不至于
                    gt_boxes_per_image = torch.empty((0, dt_boxes_per_image.shape[1]),
                                                     dtype=dt_boxes_per_image.dtype,
                                                     layout=dt_boxes_per_image.layout,
                                                     device=dt_boxes_per_image.device)

                pad = dt_boxes_per_image.shape[1] - gt_boxes_per_image.shape[1]
                # from mmcv.runner import get_dist_info
                # if get_dist_info()[0] == 0:
                #     import pdb
                #     pdb.set_trace()
                if pad > 0:
                    # gt_boxes_per_image: torch.Size([4, 4])
                    gt_boxes_per_image = F.pad(gt_boxes_per_image, (0, pad, 0, 0), value=0)  # 变成4 5 ，最后一维度是0
                    gt_boxes_per_image[:, 4] = 1  # score to be 1, std to be 0
                # pdb.set_trace()
                # if gt_boxes_per_image.shape[1] != dt_boxes_per_image[unlabel_idxs].shape[1] or len(gt_labels_per_image.shape) != len(dt_labels_per_image[unlabel_idxs].shape):
                #     pdb.set_trace()


                new_gt_bboxes.append(torch.cat([gt_boxes_per_image,
                                                dt_boxes_per_image[unlabel_idxs]]))
                if IGNORE:
                    good_dt = dt_labels_per_image[unlabel_idxs]
                    good_dt = good_dt + CLASS_ADD
                    new_gt_labels.append(torch.cat([gt_labels_per_image,
                                                    good_dt]))
                    # if good_dt.shape[0] != 0:
                    #     pdb.set_trace()
                    # 所有补框，类别+100，以便后续处理
                else:
                    new_gt_labels.append(torch.cat([gt_labels_per_image,
                                                    dt_labels_per_image[unlabel_idxs]]))
                if DEBUG and torch.sum(unlabel_idxs) > 0:
                    print("Add", sum(unlabel_idxs.int()), dt_labels_per_image[unlabel_idxs])
                    print("IoF:", ((iof_matrix * class_filter).T)[unlabel_idxs].max(1)[0])
                    print("IoB:", ((iob_matrix * class_filter).T)[unlabel_idxs].max(1)[0])
                    print("IoU:", ((iou_matrix * class_filter).T)[unlabel_idxs].max(1)[0])
                    print("Scores:", dt_boxes_per_image[unlabel_idxs][:, 4])

            return new_gt_bboxes, new_gt_labels

    def forward_teacher(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            # -----------------------------------得改-------------------------------------------------------

            teacher_info = self.extract_teacher_info(
                teacher_data["img"][torch.Tensor(tidx).to(teacher_data["img"].device).long()],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                   and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = {
            "img_metas": student_data["img_metas"],
            "transform_matrix": [
                torch.from_numpy(meta["transform_matrix"]).float().to(student_data["img"].device)
                for meta in student_data["img_metas"]
            ]}
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )
        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]


        if self.train_cfg.save_results and self.current_iter % 1 == 0:

            pseudo_bboxes_ori = self._transform_bbox(teacher_info["det_bboxes"],
                                                     [m.inverse() for m in teacher_info["transform_matrix"]],
                                                     [meta['ori_shape'] for meta in student_info["img_metas"]])
            pseudo_bboxes_cali = self._transform_bbox(teacher_info["det_bboxes_cali"],
                                                     [m.inverse() for m in teacher_info["transform_matrix"]],
                                                     [meta['ori_shape'] for meta in student_info["img_metas"]])





            gt_bboxes_ori = self._transform_bbox(student_data['gt_bboxes'],
                                                 [m.inverse() for m in student_info["transform_matrix"]],
                                                 [meta['ori_shape'] for meta in student_info["img_metas"]])
            file_paths = [meta['filename'] for meta in student_info["img_metas"]]
            gt_labels = student_data['gt_labels']
            save_dir = Path(self.train_cfg.save_dir)
            # 每张图片的路径 + 迭代次数作为 key, 每 1k 张图片存储一个文件。
            for f, pb, pl, gb, gl, pb_cali in zip(file_paths, pseudo_bboxes_ori, pseudo_labels, gt_bboxes_ori, gt_labels, pseudo_bboxes_cali):
                save_dict = {}
                save_dict['pseudo_bboxes'] = pb.detach().cpu().numpy()
                save_dict['pseudo_bboxes_cali'] = pb_cali.detach().cpu().numpy()
                save_dict['pseudo_labels'] = pl.detach().cpu().numpy()
                save_dict['gt_bboxes'] = gb.detach().cpu().numpy()
                save_dict['gt_labels'] = gl.detach().cpu().numpy()
                save_dict['iter'] = self.current_iter
                key = f + str(self.current_iter)
                if key not in self.save_dict:
                    self.save_dict[key] = save_dict

                if len(self.save_dict) == self.train_cfg.save_size:
                    save_path = save_dir / (f'{self.current_iter:07d}_' + f.replace('/', '_'))
                    mmcv.dump(self.save_dict, save_path.with_suffix('.pkl'))
                    self.save_dict = {}


        if ifcalibrate:
            cali_bboxes = self._transform_bbox(
                teacher_info["det_bboxes_cali"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )

            return pseudo_bboxes, pseudo_labels, cali_bboxes
        else:
            return pseudo_bboxes, pseudo_labels

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        # print('extract_teacher_info')
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        # tuple,len=5,每个都是bsz,256,h,w
        teacher_info["backbone_feature"] = feat
        # 当retinanet时候，必然没有proposal
        # teacher_info["proposals"] = None # 还要不要这个key？
        # 下面其实就是提取proposal的过程，为了方便起见，复用代码，将det_box也叫作proposal
        # 注意单阶段双阶段的simple_test定义不一样
        # proposal_list, proposal_label_list = self.teacher.bbox_head.simple_test_bboxes(feat, img_metas, rescale=False)
        result_list = self.teacher.bbox_head.simple_test_bboxes(feat, img_metas, rescale=False)
        # list,len=1,每个都是tuple，分别是size(0,5)和size(0)

        proposal_list, proposal_label_list = [], []
        for result in result_list:
            proposal_list.append(result[0])
            proposal_label_list.append(result[1])
        # proposal_list: list,len=1,每个是个torch.Size([100, 5])
        # proposal_label_list: list,len=1,每个是个torch.Size([0])

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        # reg_unc = self.compute_uncertainty_with_aug(
        #     feat, img_metas, proposal_list, proposal_label_list
        # )
        # det_bboxes = [
        #     torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        # ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        # pdb.set_trace()

        # --------------------------------------------start calibrate ------------------------------------------
        if ifcalibrate:
            if not self.finished_fit:
                # print(self.current_iter,'unfit, so skip')
                teacher_info["det_bboxes_cali"] = teacher_info["det_bboxes"]
            else:
                det_bboxes_uncali = copy.deepcopy(teacher_info["det_bboxes"]) # tuple:# len=1(应该是bsz) 每个都是torch.Size([0, 5])
                # det_labels = copy.deepcopy(teacher_info["det_labels"])
                current_device = img.device
                # pdb.set_trace()

                for i in range(len(det_bboxes_uncali)):
                    det_bboxes_uncali_perimg = det_bboxes_uncali[i] # tensor
                    det_confidence_uncali_perimg = det_bboxes_uncali_perimg[:,-1].cpu().numpy()

                    if det_confidence_uncali_perimg.shape[0] == 0:
                        pass
                    else:
                        # pdb.set_trace()
                        det_confidence_cali_perimg = self.calibrator.transform(det_confidence_uncali_perimg) # np 不能len也不能shape
                        det_confidence_cali_perimg = torch.from_numpy(det_confidence_cali_perimg).to(current_device) # tensor
                        # # num_ini_box = det_confidence_cali_perimg.shape[0]
                        # det_index_after_cali = det_confidence_cali_perimg >= self.thr_after_cali
                        # print('shape is', det_confidence_cali_perimg.shape)
                        #
                        # det_bboxes_uncali[i] = det_bboxes_uncali[i][det_index_after_cali]
                        det_bboxes_uncali[i][:, -1] = det_confidence_cali_perimg
                        # det_labels[i] = det_labels[i][det_index_after_cali]
                        #
                        #
                        # print('and', det_confidence_cali_perimg.shape)
                        # num_cali_box = det_confidence_cali_perimg.shape[0]
                        # print('during calibration, we drop num box', num_cali_box-num_ini_box)

                        # det_bboxes_uncali[i][:,-1] = torch.from_numpy(det_confidence_cali_perimg).to(current_device)

                teacher_info["det_bboxes_cali"] = det_bboxes_uncali
                # teacher_info["det_labels"] = det_labels

        # --------------------------------------------end calibrate ------------------------------------------




        return teacher_info

    def compute_uncertainty_with_aug(
            self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                    torch.randn(times, box.shape[0], 4, device=box.device)
                    * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
