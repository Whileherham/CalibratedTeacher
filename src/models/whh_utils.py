import pdb
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from netcal.scaling import LogisticCalibration
import torch
import pickle


def calibrate(dt_boxes_per_image, dt_labels_per_image, calibrator, tp_thre=0.5,iter=0):
    """

    Args:
        dt_boxes_per_image: torch.Size([0, 5])
        dt_labels_per_image: torch.Size([0])
        calibrator:

    Returns:

    """
    # if dt_boxes_per_image.shape[1] != 5:
    #     print('NNN')
    #     pdb.set_trace()


    if dt_labels_per_image.shape[0] == 0:
        print('nothing to calibrate, skip!')
        return dt_boxes_per_image, dt_labels_per_image

    # if dt_labels_per_image.device == torch.device('cpu'):
    #     dt_boxes_per_image, dt_labels_per_image = dt_boxes_per_image.numpy(), dt_labels_per_image.numpy()
    # else:
    #     dt_boxes_per_image, dt_labels_per_image = dt_boxes_per_image.cpu().numpy(), dt_labels_per_image.cpu().numpy()

    input = dt_boxes_per_image[:, -1]

    if False in input >= 0 or False in input <= 1:
        print('not all in [0,1], skip')
        return dt_boxes_per_image, dt_labels_per_image


    if dt_labels_per_image.device == torch.device('cpu'):
        input = input.numpy()
    else:
        input = input.cpu().numpy()

    # pdb.set_trace()
    calibrated = torch.from_numpy(calibrator.transform(input))

    index_tp = calibrated >= tp_thre
    dt_boxes_per_image = dt_boxes_per_image[index_tp]
    dt_labels_per_image = dt_labels_per_image[index_tp]

    # print('not finish')
    if len(dt_boxes_per_image.shape) != 2:
        print('dt_boxes_per_image,wrong',dt_boxes_per_image.shape)
        # pdb.set_trace()
        dt_boxes_per_image = dt_boxes_per_image[0]
        # 奇怪的现象MMP，如果全是true， 那么维度升高，例如a=[1]，a[true]。shape= 1,1
        # pdb.set_trace()
    if len(dt_labels_per_image.shape) != 1:
        print('dt_labels_per_image,wrong',dt_labels_per_image.shape)
        # pdb.set_trace()
        dt_labels_per_image = dt_labels_per_image[0]

    return dt_boxes_per_image, dt_labels_per_image


def match_per_iter(pseudo_bboxes, pseudo_labels, gt_bboxes, gt_labels, tp_thre = 0.5):
    """

    Args:
        pseudo_bboxes: list len=1 torch.Size([0, 5])
        pseudo_labels: ...........torch.Size([0])
        gt_bboxes: ...............torch.Size([10, 4])
        gt_labels: ...............torch.Size([10])
        tp_thre: tp的iou阈值

    Returns:
        match_iter
        confidence_iter

    """
    match_iter = []
    confidence_iter = []

    for gt_boxes_per_image, gt_labels_per_image, dt_boxes_per_image, dt_labels_per_image in zip(
            gt_bboxes, gt_labels, pseudo_bboxes, pseudo_labels):

        # if dt_labels_per_image.shape[0] == 0: # 太惨了，一个高置信度框框都没有，直接跳过
        #     continue

        target_class_list = gt_labels_per_image.reshape(-1, 1)
        pred_class_list = dt_labels_per_image.reshape(1, -1)
        class_filter = target_class_list == pred_class_list


        iou_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4])

        iou_match = (iou_matrix > tp_thre) & class_filter # gt,dt
        # 这里逻辑跟选框不一样，这里iou_match 为TRUE表示检测框跟gt匹配上了，这些是要保留，而非之后过滤
        # iou_filter = (iou_matrix > tp_thre) & class_filter

        tp_index = torch.sum(iou_match, 0) > 0 # len=dt, TRUE表示这个框匹配上了

        # unlabel_idxs = torch.sum(final_filter, 0) == 0  # whh: TRUE表示剩下的bbox的ids len=dt
        match_iter.append(tp_index.unsqueeze(0))
        confidence_iter.append(dt_boxes_per_image[:,-1].unsqueeze(0))



    # pdb.set_trace()
    match_iter = torch.cat(match_iter, dim=1) # 1 dt-all
    confidence_iter = torch.cat(confidence_iter, dim=1)
    return match_iter, confidence_iter


def get_sparse_per_iter(pseudo_bboxes, pseudo_labels, gt_bboxes, gt_labels, tp_thre = 0.5):
    """
    与上一个相比，这个函数的目的是，首先按照亮哥选框的逻辑，把bbox里面和gt有一定重合的那些框框拿出来
    用这部分框，跟稀疏gt，来拟合校正
    但是显然，这时候准确率偏高，因为那些FN在这里全部被舍去了，所以，建议此时匹配用更苛刻的准则
    Args:
        pseudo_bboxes:
        pseudo_labels:
        gt_bboxes:
        gt_labels:
        tp_thre:

    Returns:

    """
    # pdb.set_trace()
    # unfinished

    pseudo_labels2 = list(pseudo_labels).copy()
    for i in range(len(pseudo_bboxes)):
        gt_boxes_per_image, gt_labels_per_image, dt_boxes_per_image, dt_labels_per_image = \
            gt_bboxes[i], gt_labels[i], pseudo_bboxes[i], pseudo_labels2[i]
        target_class_list = gt_labels_per_image.reshape(-1, 1)
        pred_class_list = dt_labels_per_image.reshape(1, -1)
        class_filter = target_class_list == pred_class_list

        iob_matrix = bbox_overlaps(dt_boxes_per_image[:, :4], gt_boxes_per_image, mode='iof').T  # [gt,dt]
        iob_filter = (iob_matrix > 0.9) & class_filter  # whh:box重合且类别对得上，认为是一个 #[gt,dt]

        iof_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4], mode='iof')  # [gt,dt]
        iof_filter = (iof_matrix > 0.9) & class_filter

        iou_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4])
        iou_filter = (iou_matrix > tp_thre) & class_filter

        final_filter = iou_filter | (iou_matrix > 0.75) | iof_filter | iob_filter  # gt dt
        # whh:这块应该是把dt和gt 大幅度相交的直接过滤掉，但是为什么要分别算俩iof，是因为只用iou不够嘛

        unlabel_idxs = torch.sum(final_filter, 0) > 0  # whh: TRUE表示可以用力匹配sparse gt的
        # pdb.set_trace()
        pseudo_bboxes[i], pseudo_labels2[i] = dt_boxes_per_image[unlabel_idxs], dt_labels_per_image[unlabel_idxs]



    pseudo_labels2 = tuple(pseudo_labels2)

    return pseudo_bboxes, pseudo_labels2


def fit_calibrator(data2fit, calibrator, method = 'logistic', input_variable = 'confidence'):
    """

    Args:
        data2fit:
            self.data2fit[i+1] = {}
            self.data2fit[i+1]['matches'] = [],每个都是【1，num] 的torch
            self.data2fit[i+1]['confidences'] = []
        calibrator:
        method:
        input_variable:

    Returns:

    """
    assert method in ['logistic']
    assert input_variable in ['confidence']

    # 构造待拟合input
    if input_variable == 'confidence':

        input_train = [data2fit[iter]['confidences'] for iter in data2fit.keys()]
        matched_train = [data2fit[iter]['matches'] for iter in data2fit.keys()]
        input_train = torch.cat(input_train, dim=1)[0] # tensor, torch.Size([n])
        matched_train = torch.cat(matched_train, dim=1)[0]
        # 防止没有输入
        if input_train.shape[0] == 0:
            # print('no input, skip')
            return calibrator, False
        if False in input_train >= 0 or False in input_train <= 1:
            print('input_train not all in [0,1]')
            return calibrator, False
        if (0 not in matched_train) or (1 not in matched_train):
            print('matched all 0 or 1 ')
            return calibrator, False


        print('we successfully fit')

        if input_train.device == torch.device('cpu'):
            input_train, matched_train = input_train.numpy(), matched_train.numpy()
        else:
            input_train, matched_train = input_train.cpu().numpy(), matched_train.cpu().numpy()

    else:
        print('not finish')
    # pdb.set_trace()

    calibrator.fit(input_train, matched_train)
    print('successfully fit')
    return calibrator, True

def filter_sparse_dt(pbs,pls,gbs,gls,iob_thr=0.9, mining_th_score=0.5, iou_second_thr=0.75, returnid=False):
    """
    把检测框的结果里，被稀疏标注舍去的那些dt删掉。舍去dt的后三个参数，跟实际代码保持一致即可

    :param pbs: 下面四个都是tensor
    :param pls:
    :param gbs:
    :param gls:
    :param iob_thr:
    :param mining_th_score:
    :param iou_second_thr:
    :return:
    """
    # 参数保持一致
    dt_boxes_per_image = pbs
    gt_boxes_per_image = gbs
    dt_labels_per_image = pls
    gt_labels_per_image = gls

    target_class_list = gt_labels_per_image.reshape(-1, 1)
    pred_class_list = dt_labels_per_image.reshape(1, -1)
    class_filter = target_class_list == pred_class_list

    iob_matrix = bbox_overlaps(dt_boxes_per_image[:, :4], gt_boxes_per_image, mode='iof').T  # [gt,dt]
    iob_filter = (iob_matrix > iob_thr) & class_filter  # whh:box重合且类别对得上，认为是一个 #[gt,dt]

    iof_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4], mode='iof')  # [gt,dt]
    iof_filter = (iof_matrix > iob_thr) & class_filter

    iou_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4])
    iou_filter = (iou_matrix > mining_th_score) & class_filter

    final_filter = iou_filter | (iou_matrix > iou_second_thr) | iof_filter | iob_filter  # gt dt
    unlabel_idxs = torch.sum(final_filter, 0) == 0

    # if dt_labels_per_image.shape[0] > 0:
    #     pdb.set_trace()

    # pdb.set_trace()
    if returnid:
        return dt_boxes_per_image[unlabel_idxs], dt_labels_per_image[unlabel_idxs], unlabel_idxs
    else:
        return dt_boxes_per_image[unlabel_idxs], dt_labels_per_image[unlabel_idxs]

# def filter_sparse_dt(pbs,pls,gbs,gls,iob_thr=0.9, mining_th_score=0.5, iou_second_thr=0.75 ):
#     """
#     把检测框的结果里，被稀疏标注舍去的那些dt删掉。舍去dt的后三个参数，跟实际代码保持一致即可
#
#     :param pbs: 下面四个都是tensor
#     :param pls:
#     :param gbs:
#     :param gls:
#     :param iob_thr:
#     :param mining_th_score:
#     :param iou_second_thr:
#     :return:
#     """
#     # 参数保持一致
#     dt_boxes_per_image = pbs
#     gt_boxes_per_image = gbs
#     dt_labels_per_image = pls
#     gt_labels_per_image = gls
#
#     target_class_list = gt_labels_per_image.reshape(-1, 1)
#     pred_class_list = dt_labels_per_image.reshape(1, -1)
#     class_filter = target_class_list == pred_class_list
#
#     iob_matrix = bbox_overlaps(dt_boxes_per_image[:, :4], gt_boxes_per_image, mode='iof').T  # [gt,dt]
#     iob_filter = (iob_matrix > iob_thr) & class_filter  # whh:box重合且类别对得上，认为是一个 #[gt,dt]
#
#     iof_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4], mode='iof')  # [gt,dt]
#     iof_filter = (iof_matrix > iob_thr) & class_filter
#
#     iou_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4])
#     iou_filter = (iou_matrix > mining_th_score) & class_filter
#
#     final_filter = iou_filter | (iou_matrix > iou_second_thr) | iof_filter | iob_filter  # gt dt
#     unlabel_idxs = torch.sum(final_filter, 0) == 0
#
#     # if dt_labels_per_image.shape[0] > 0:
#     #     pdb.set_trace()
#
#     # pdb.set_trace()
#
#     return dt_boxes_per_image[unlabel_idxs], dt_labels_per_image[unlabel_idxs]

def split_sparse_dt(pbs,pls,gbs,gls,iob_thr=0.9, mining_th_score=0.5, iou_second_thr=0.75,returnid=False ):
    """
    与filter_sparse_dt不同点在于，这里留下的是匹配到稀疏的那些

    :param pbs: 下面四个都是tensor
    :param pls:
    :param gbs:
    :param gls:
    :param iob_thr:
    :param mining_th_score:
    :param iou_second_thr:
    :return:
    """
    # 参数保持一致
    dt_boxes_per_image = pbs
    gt_boxes_per_image = gbs
    dt_labels_per_image = pls
    gt_labels_per_image = gls

    target_class_list = gt_labels_per_image.reshape(-1, 1)
    pred_class_list = dt_labels_per_image.reshape(1, -1)
    class_filter = target_class_list == pred_class_list

    iob_matrix = bbox_overlaps(dt_boxes_per_image[:, :4], gt_boxes_per_image, mode='iof').T  # [gt,dt]
    iob_filter = (iob_matrix > iob_thr) & class_filter  # whh:box重合且类别对得上，认为是一个 #[gt,dt]

    iof_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4], mode='iof')  # [gt,dt]
    iof_filter = (iof_matrix > iob_thr) & class_filter

    iou_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4])
    iou_filter = (iou_matrix > mining_th_score) & class_filter

    final_filter = iou_filter | (iou_matrix > iou_second_thr) | iof_filter | iob_filter  # gt dt
    unlabel_idxs = torch.sum(final_filter, 0) == 0

    # if dt_labels_per_image.shape[0] > 0:
    #     pdb.set_trace()

    # pdb.set_trace()
    match_sparse_id = unlabel_idxs == 0
    # pdb.set_trace()
    if returnid:
        return dt_boxes_per_image[match_sparse_id], dt_labels_per_image[match_sparse_id], match_sparse_id
    else:

        return dt_boxes_per_image[match_sparse_id], dt_labels_per_image[match_sparse_id]


# def split_sparse_dt(pbs,pls,gbs,gls,iob_thr=0.9, mining_th_score=0.5, iou_second_thr=0.75 ):
#     """
#     与filter_sparse_dt不同点在于，这里留下的是匹配到稀疏的那些
#
#     :param pbs: 下面四个都是tensor
#     :param pls:
#     :param gbs:
#     :param gls:
#     :param iob_thr:
#     :param mining_th_score:
#     :param iou_second_thr:
#     :return:
#     """
#     # 参数保持一致
#     dt_boxes_per_image = pbs
#     gt_boxes_per_image = gbs
#     dt_labels_per_image = pls
#     gt_labels_per_image = gls
#
#     target_class_list = gt_labels_per_image.reshape(-1, 1)
#     pred_class_list = dt_labels_per_image.reshape(1, -1)
#     class_filter = target_class_list == pred_class_list
#
#     iob_matrix = bbox_overlaps(dt_boxes_per_image[:, :4], gt_boxes_per_image, mode='iof').T  # [gt,dt]
#     iob_filter = (iob_matrix > iob_thr) & class_filter  # whh:box重合且类别对得上，认为是一个 #[gt,dt]
#
#     iof_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4], mode='iof')  # [gt,dt]
#     iof_filter = (iof_matrix > iob_thr) & class_filter
#
#     iou_matrix = bbox_overlaps(gt_boxes_per_image, dt_boxes_per_image[:, :4])
#     iou_filter = (iou_matrix > mining_th_score) & class_filter
#
#     final_filter = iou_filter | (iou_matrix > iou_second_thr) | iof_filter | iob_filter  # gt dt
#     unlabel_idxs = torch.sum(final_filter, 0) == 0
#
#     # if dt_labels_per_image.shape[0] > 0:
#     #     pdb.set_trace()
#
#     # pdb.set_trace()
#     match_sparse_id = unlabel_idxs == 0
#     # pdb.set_trace()
#
#     return dt_boxes_per_image[match_sparse_id], dt_labels_per_image[match_sparse_id]

def load_from_8pkls(pkl_files_8,saved_path,tp_iou_thre_sparse=0.75):
    confidences_8_sparse = []
    confidences_8_left = []
    matches_8_sparse = []
    # matches_8_left = []
    num_gt_bbox_sparse = 0
    num_dt_bbox_all = 0
    for pkl_file in pkl_files_8:

        with open(saved_path + pkl_file, 'rb') as f:
            records = pickle.load(f)  # list
            for k in records.keys():
                # k表示一张图的结果 例如k=data/coco/train2017/000000115113.jpg1
                result = records[k]
                # gbs_full, gls_full = get_full_anno(k)  # full-bbox-gt

                # 预测的bbox，和稀疏gt
                pbs_sparse = torch.from_numpy(result['pseudo_bboxes'])
                pls_sparse = torch.from_numpy(result['pseudo_labels'])
                gbs_sparse = torch.from_numpy(result['gt_bboxes'])  # torch.tensor([1,4])
                gls_sparse = torch.from_numpy(result['gt_labels'])
                num_gt_bbox_sparse += gls_sparse.shape[0]
                num_dt_bbox_all += pls_sparse.shape[0]

                # 按照亮哥代码准则，根据稀疏gt舍去部分dt,left表示剩余dt,补框从这里补
                pbs_left, pls_left = filter_sparse_dt(pbs_sparse, pls_sparse, gbs_sparse, gls_sparse)

                # 这里sparse表示除去那些miss部分的dt,也就是跟稀疏部分匹配的那些dt
                pbs_sparse, pls_sparse = split_sparse_dt(pbs_sparse, pls_sparse, gbs_sparse, gls_sparse)

                if pbs_left.shape[0] != 0:
                    confidences_8_left.append(pbs_left[:, -1].unsqueeze(0))
                if pbs_sparse.shape[0] != 0:  #
                    target_class_list_sparse = gls_sparse.reshape(-1, 1)
                    pred_class_list_sparse = pls_sparse.reshape(1, -1)  # 仅用于拟合
                    class_filter_sparse = target_class_list_sparse == pred_class_list_sparse  # 2 1  #n_g n_p

                    iou_matrix_sparse = bbox_overlaps(gbs_sparse, pbs_sparse[:, :4])  # n_g n_p

                    match_sparse = (iou_matrix_sparse > tp_iou_thre_sparse) * class_filter_sparse  # 如果
                    match_sparse = torch.sum(match_sparse, dim=0) > 0  # tensor np
                    matches_8_sparse.append((match_sparse * 1.0).unsqueeze(0))
                    confidences_8_sparse.append(pbs_sparse[:, -1].unsqueeze(0))


    if matches_8_sparse == []:
        return [], [], [], []
    else:
        matches_8_sparse = torch.cat(matches_8_sparse, dim=1)[0] # torch.Size([num])
        confidences_8_sparse = torch.cat(confidences_8_sparse, dim=1)[0] # torch.Size([num])

        if confidences_8_left == []:
            return matches_8_sparse,[],confidences_8_sparse,confidences_8_left
        else:
            confidences_8_left = torch.cat(confidences_8_left, dim=1)[0] # torch.Size([num2])
            pseudo_matches_8_left = confidences_8_left.clone() * 0.0
            # 这是个假的，只是为了方便画图
            return matches_8_sparse,pseudo_matches_8_left,confidences_8_sparse,confidences_8_left


def load_from_8pkls_doublecon(pkl_files_8,saved_path,tp_iou_thre_sparse=0.75):
    bboxconfidences_8_sparse = []
    bboxconfidences_8_left = []
    confidences_8_sparse = []
    confidences_8_left = []
    matches_8_sparse = []
    # matches_8_left = []
    num_gt_bbox_sparse = 0
    num_dt_bbox_all = 0
    for pkl_file in pkl_files_8:

        with open(saved_path + pkl_file, 'rb') as f:
            records = pickle.load(f)  # list
            for k in records.keys():
                # k表示一张图的结果 例如k=data/coco/train2017/000000115113.jpg1
                result = records[k]
                # gbs_full, gls_full = get_full_anno(k)  # full-bbox-gt

                # 预测的bbox，和稀疏gt
                pbs_sparse = torch.from_numpy(result['pseudo_bboxes'])
                pls_sparse = torch.from_numpy(result['pseudo_labels'])
                gbs_sparse = torch.from_numpy(result['gt_bboxes'])  # torch.tensor([1,4])
                gls_sparse = torch.from_numpy(result['gt_labels'])
                bboxconfidence_sparse = torch.from_numpy(result['det_bbox_confidence'])


                num_gt_bbox_sparse += gls_sparse.shape[0]
                num_dt_bbox_all += pls_sparse.shape[0]

                # 按照亮哥代码准则，根据稀疏gt舍去部分dt,left表示剩余dt,补框从这里补
                # pbs_left, pls_left, leftid = filter_sparse_dt(pbs_sparse, pls_sparse, gbs_sparse, gls_sparse)
                pbs_left, pls_left, leftid = filter_sparse_dt(pbs_sparse, pls_sparse, gbs_sparse, gls_sparse,
                                                              returnid=True)
                bboxconfidence_left = bboxconfidence_sparse[leftid]
                assert bboxconfidence_left.shape[0] == pls_left.shape[0]

                # 这里sparse表示除去那些miss部分的dt,也就是跟稀疏部分匹配的那些dt
                # pbs_sparse, pls_sparse = split_sparse_dt(pbs_sparse, pls_sparse, gbs_sparse, gls_sparse)
                pbs_sparse, pls_sparse, sparseid = split_sparse_dt(pbs_sparse, pls_sparse, gbs_sparse, gls_sparse,
                                                                   returnid=True)
                bboxconfidence_sparse = bboxconfidence_sparse[sparseid]
                assert bboxconfidence_sparse.shape[0] == pls_sparse.shape[0]




                if pbs_left.shape[0] != 0:
                    confidences_8_left.append(pbs_left[:, -1].unsqueeze(0))
                    bboxconfidences_8_left.append(bboxconfidence_left.unsqueeze(0))
                if pbs_sparse.shape[0] != 0:  #
                    target_class_list_sparse = gls_sparse.reshape(-1, 1)
                    pred_class_list_sparse = pls_sparse.reshape(1, -1)  # 仅用于拟合
                    class_filter_sparse = target_class_list_sparse == pred_class_list_sparse  # 2 1  #n_g n_p

                    iou_matrix_sparse = bbox_overlaps(gbs_sparse, pbs_sparse[:, :4])  # n_g n_p

                    match_sparse = (iou_matrix_sparse > tp_iou_thre_sparse) * class_filter_sparse  # 如果
                    match_sparse = torch.sum(match_sparse, dim=0) > 0  # tensor np
                    matches_8_sparse.append((match_sparse * 1.0).unsqueeze(0))
                    confidences_8_sparse.append(pbs_sparse[:, -1].unsqueeze(0))
                    bboxconfidences_8_sparse.append(bboxconfidence_sparse.unsqueeze(0))


    if matches_8_sparse == []:
        return [], [], [], [], [], []
    else:
        matches_8_sparse = torch.cat(matches_8_sparse, dim=1)[0] # torch.Size([num])
        confidences_8_sparse = torch.cat(confidences_8_sparse, dim=1)[0] # torch.Size([num])
        bboxconfidences_8_sparse = torch.cat(bboxconfidences_8_sparse, dim=1)[0]

        if confidences_8_left == []:
            return matches_8_sparse,[],confidences_8_sparse,confidences_8_left,\
                   bboxconfidences_8_sparse,bboxconfidences_8_left
        else:
            confidences_8_left = torch.cat(confidences_8_left, dim=1)[0] # torch.Size([num2])
            bboxconfidences_8_left = torch.cat(bboxconfidences_8_left, dim=1)[0]
            pseudo_matches_8_left = confidences_8_left.clone() * 0.0
            # 这是个假的，只是为了方便画图
            return matches_8_sparse,pseudo_matches_8_left,confidences_8_sparse,confidences_8_left,\
                   bboxconfidences_8_sparse,bboxconfidences_8_left





# if __name__ == '__main__':
#     import os
#     path_pkl = '/youtu_fuxi_team1_ceph/haohanwang/xishu/xishu_debug/tempcalifcos0/save'
#     for root, ds, files in os.walk(path_pkl):
#         files.sort(reverse=False)
#         l = 0  # 500iter 0
#         index_first = l * 8
#         index_last = index_first + 8
#         pkl_files_8 = files[index_first:index_last]
#         load_from_8pkls(pkl_files_8=pkl_files_8,
#                         saved_path='/youtu_fuxi_team1_ceph/haohanwang/xishu/xishu_debug/tempcalifcos0/save/')