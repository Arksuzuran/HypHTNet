from os import path
import os
import numpy as np
import cv2
import time
from datetime import datetime
import logging
import pandas
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
import torch
from model import HTNet
import numpy as np
from facenet_pytorch import MTCNN
from utils import LossFunction, Logger, filter_filename_by_datatype

# 数据集及其对应的编号
datatype_dic = {'0': 'samm', '1': 'smic', '2': 'casme2', 'all': 'all'}


# Some of the codes are adapted from STSNet
def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''


def whole_face_block_coordinates(dataset_type):
    """
    get the whole face block coordinates
    :param dataset_type: 指定的数据集
    """
    # 提取指定数据集的数据
    raw_df = pandas.read_csv('combined_3_class2_for_optical_flow.csv')
    if dataset_type == 'all':
        df = raw_df
    else:
        df = raw_df[raw_df['dataset'] == datatype_dic[dataset_type]].copy().reset_index(drop=True)
    # print("whole_face_block_coordinates提取数据", df)

    m, n = df.shape
    base_data_src = './datasets/combined_datasets_whole'
    total_emotion = 0
    image_size_u_v = 28
    # get the block center coordinates
    face_block_coordinates = {}

    # for i in range(0, m):
    for i in range(0, m):
        image_name = str(df['sub'][i]) + '_' + str(
            df['filename_o'][i]) + ' .png'
        # print(image_name)
        img_path_apex = base_data_src + '/' + df['imagename'][i]
        train_face_image_apex = cv2.imread(img_path_apex)  # (444, 533, 3)
        face_apex = cv2.resize(train_face_image_apex, (28, 28), interpolation=cv2.INTER_AREA)
        # get face and bounding box
        mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
        batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)
        # print(img_path_apex,batch_landmarks)
        # if not detecting face
        if batch_landmarks is None:
            # print( df['imagename'][i])
            batch_landmarks = np.array([[[9.528073, 11.062551]
                                            , [21.396168, 10.919773]
                                            , [15.380184, 17.380562]
                                            , [10.255435, 22.121233]
                                            , [20.583706, 22.25584]]])
            # print(img_path_apex)
        row_n, col_n = np.shape(batch_landmarks[0])
        # print(batch_landmarks[0])
        for i in range(0, row_n):
            for j in range(0, col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21
        batch_landmarks = batch_landmarks.astype(int)
        # print(batch_landmarks[0])
        # get the block center coordinates
        face_block_coordinates[image_name] = batch_landmarks[0]
    # print(len(face_block_coordinates))
    return face_block_coordinates


def crop_optical_flow_block(dataset_type):
    """
    crop the 28*28-> 14*14 according to i5 image centers
    :param dataset_type: 指定的数据集
    """
    face_block_coordinates_dict = whole_face_block_coordinates(dataset_type)
    # print(len(face_block_coordinates_dict))
    # Get train dataset
    whole_optical_flow_path = './datasets/STSNet_whole_norm_u_v_os'
    whole_optical_flow_imgs = filter_filename_by_datatype(whole_optical_flow_path, dataset_type)
    four_parts_optical_flow_imgs = {}
    # print(whole_optical_flow_imgs[0]) #spNO.189_f_150.png
    for n_img in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[n_img] = []
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        four_part_coordinates = face_block_coordinates_dict[n_img]
        # 每个部位都是14 * 14
        l_eye = flow_image[four_part_coordinates[0][0] - 7:four_part_coordinates[0][0] + 7,
                four_part_coordinates[0][1] - 7: four_part_coordinates[0][1] + 7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7:four_part_coordinates[1][0] + 7,
                 four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        nose = flow_image[four_part_coordinates[2][0] - 7:four_part_coordinates[2][0] + 7,
               four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        r_eye = flow_image[four_part_coordinates[3][0] - 7:four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7:four_part_coordinates[4][0] + 7,
                 four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)
        # print(np.shape(l_eye))
    # print((four_parts_optical_flow_imgs['spNO.189_f_150.png'][0]))->(14,14,3)
    print(len(four_parts_optical_flow_imgs))
    return four_parts_optical_flow_imgs


class Fusionmodel(nn.Module):
    def __init__(self):
        #  extend from original
        super(Fusionmodel, self).__init__()
        self.fc1 = nn.Linear(15, 3)  # 15->3
        self.bn1 = nn.BatchNorm1d(3)
        self.d1 = nn.Dropout(p=0.5)
        # Linear 256 to 26
        self.fc_2 = nn.Linear(6, 2)  # 6->3
        # self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

        # forward layers is to use these layers above

    def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
        fuse_five_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
        # nn.linear - fc
        fuse_out = self.fc1(fuse_five_features)
        # fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)  # drop out
        #
        fuse_whole_five_parts = torch.cat(
            (whole_feature, fuse_out), 0)
        # fuse_whole_five_parts = self.bn1(fuse_whole_five_parts)
        fuse_whole_five_parts = self.relu(fuse_whole_five_parts)
        fuse_whole_five_parts = self.d1(fuse_whole_five_parts)  # drop out
        out = self.fc_2(fuse_whole_five_parts)
        return out


def main(config):
    learning_rate = 0.00005
    batch_size = 256
    patch_size = config.psz
    epochs = 800
    alpha = config.a
    metric = config.m
    all_accuracy_dict = {}

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset_type = config.d  # 本次eval所选的数据集
    exp_time = datetime.now().strftime('%m-%d-%H-%M-%S')
    main_dataset_dir = './datasets/three_norm_u_v_os'
    weight_dir = './ourmodel_threedatasets_weights/' + exp_time
    if (config.train):
        logger = Logger('./log/', f'training--{exp_time}--a{alpha}--m{metric}.log')
        os.makedirs(weight_dir, exist_ok=True)  # 训练时创建权重目录
    else:
        logger = Logger('./log/', f'eval--{datatype_dic[dataset_type]}--w{config.wdir}--{exp_time}.log')
        weight_dir = './ourmodel_threedatasets_weights/' + config.wdir  # 测试时需要选择加载指定的权重目录
        epochs = 1  # 测试时无需多个epochs

    logger('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))
    logger(vars(config))

    loss_fn = LossFunction(alpha=alpha, metric=metric)
    total_gt = []
    total_pred = []
    best_total_pred = []

    t = time.time()

    # get data by dataset type
    if (config.train):
        sub_name = filter_filename_by_datatype(main_dataset_dir, "all")
    else:
        sub_name = filter_filename_by_datatype(main_dataset_dir, dataset_type)
    all_five_parts_optical_flow = crop_optical_flow_block("all")
    logger(sub_name)

    for n_subName in sub_name:
        logger(f'==============Subject: {n_subName}==============')
        y_train = []
        y_test = []
        four_parts_train = []
        four_parts_test = []

        # Get train dataset
        expression = os.listdir(main_dataset_dir + '/' + n_subName + '/u_train')
        for n_expression in expression:
            img = os.listdir(main_dataset_dir + '/' + n_subName + '/u_train/' + n_expression)

            for n_img in img:
                y_train.append(int(n_expression))
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_train.append(lr_eye_lips)

        # Get test dataset
        expression = os.listdir(main_dataset_dir + '/' + n_subName + '/u_test')
        for n_expression in expression:
            img = os.listdir(main_dataset_dir + '/' + n_subName + '/u_test/' + n_expression)

            for n_img in img:
                y_test.append(int(n_expression))
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_test.append(lr_eye_lips)
        weight_path = weight_dir + '/' + n_subName + '.pth'

        # Reset or load model weigts
        model = HTNet(
            image_size=28,
            patch_size=patch_size,
            dim=256,  # 256,--96, 56-, 192
            heads=3,  # 3 ---- , 6-
            num_hierarchies=3,  # 3----number of hierarchies
            block_repeats=(2, 2, 10),  # (2, 2, 8),------
            # the number of transformer blocks at each heirarchy, starting from the bottom(2,2,20) -
            num_classes=3
        )

        model = model.to(device)
        # if torch.cuda.device_count() > 1:
        #     logger(f"Using {torch.cuda.device_count()} GPUs...")
        #     model = nn.DataParallel(model)

        # 训练 or 加载模型参数
        if (config.train):
            # model.apply(reset_weights)
            logger('Train')
        else:
            model.load_state_dict(torch.load(weight_path))
            logger('Eval\nLoad model weights: ' + weight_path)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        four_parts_train = torch.Tensor(np.array(four_parts_train)).permute(0, 3, 1, 2)
        dataset_train = TensorDataset(four_parts_train, y_train)
        train_dl = DataLoader(dataset_train, batch_size=batch_size)

        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test = torch.Tensor(np.array(four_parts_test)).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(four_parts_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size)

        # store best results
        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, epochs + 1):
            if (config.train):
                # Training
                model.train()
                losses = {'train_loss': 0.0, 'cls_loss': 0.0, 'cst_loss': 0.0}  # 总损失 分类损失 对比损失
                # train_loss = 0.0

                num_train_correct = 0
                num_train_examples = 0

                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    x_p, y_hat = model(x)

                    # TODO:引入对比学习
                    loss, cls_loss, cst_loss = loss_fn(x_p, y_hat, y)
                    loss.backward()
                    optimizer.step()

                    for name, loss_value in zip(losses.keys(), [loss, cls_loss, cst_loss]):
                        losses[name] += loss_value.data.item() * x.size(0)
                    # train_loss += loss.data.item() * x.size(0)

                    num_train_correct += (torch.max(y_hat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                train_acc = num_train_correct / num_train_examples
                for name in losses.keys():
                    losses[name] = losses[name] / len(train_dl.dataset)
                # train_loss = train_loss / len(train_dl.dataset)

                if epoch % 10 == 0:
                    logger(f'Epoch [{epoch}], \tTrain Acc: {train_acc:.8f}, \tTrain Loss:  {losses["train_loss"]:.8f},'
                           f'\tClassification Loss: {losses["cls_loss"]:.8f}, \tContrastive Loss: {losses["cst_loss"]:.8f}')

            # Testing
            model.eval()
            losses = {'val_loss': 0.0, 'cls_loss': 0.0, 'cst_loss': 0.0}  # 总损失 分类损失 对比损失
            num_val_correct = 0
            num_val_examples = 0
            for batch in test_dl:
                x = batch[0].to(device)
                y = batch[1].to(device)
                x_p, y_hat = model(x)
                loss, cls_loss, cst_loss = loss_fn(x_p, y_hat, y)
                for name, loss_value in zip(losses.keys(), [loss, cls_loss, cst_loss]):
                    losses[name] += loss_value.data.item() * x.size(0)
                num_val_correct += (torch.max(y_hat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            for name in losses.keys():
                losses[name] = losses[name] / len(train_dl.dataset)
            if epoch % 10 == 0:
                logger(f'Epoch [{epoch}], \tVal Acc: {val_acc:.8f}, \tVal Loss:  {losses["val_loss"]:.8f},'
                       f'\t\tClassification Loss: {losses["cls_loss"]:.8f}, \tContrastive Loss: {losses["cst_loss"]:.8f}')

            #### best result
            temp_best_each_subject_pred = []
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                temp_best_each_subject_pred.extend(torch.max(y_hat, 1)[1].tolist())
                best_each_subject_pred = temp_best_each_subject_pred
                # Save Weights
                if (config.train):
                    torch.save(model.state_dict(), weight_path)

        # For UF1 and UAR computation
        logger(f'Best Predicted:\t{best_each_subject_pred}')
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = y.tolist()
        all_accuracy_dict[n_subName] = accuracydict

        logger(f'Ground Truth :\t{y.tolist()}')
        logger(f'Evaluation until this subject: ')
        total_pred.extend(torch.max(y_hat, 1)[1].tolist())
        total_gt.extend(y.tolist())
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        logger(f'best UF1: {round(best_UF1, 4)} \t best UAR: {round(best_UAR, 4)}')

    if(config.train):
        logger('Final Evaluation on training: ')
    else:
        logger(f'Final Evaluation on {datatype_dic[dataset_type]}: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    logger(np.shape(total_gt))
    logger(f'Total Time Taken: {time.time() - t}')
    logger(all_accuracy_dict)


if __name__ == '__main__':
    # get_whole_u_v_os()
    # create_norm_u_v_os_train_test()
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--train', type=strtobool, default=False)  # Train or use pre-trained weight for prediction
    parser.add_argument('--a', type=float, default='0.001',
                        help='Weight of contrastive loss')
    parser.add_argument('--m', type=str, default='p', help='Metric used for loss function. e for Euclidean distance, '
                                                           'p for normal Hyperbolic distance, '
                                                           'd for Hyperbolic dot produce distance.'
                                                           '(default: p)')
    parser.add_argument('--d', type=str, default='all', help='Dataset to eval the model. Only work when --train '
                                                             'False. 0 for SAMM, 1 for SMIC , 2 for CASMEII,'
                                                             'default for combined dataset')
    parser.add_argument('--wdir', type=str, default='01-19-00-50-12', help='Model weight dir to load(01-19-00-50-12). '
                                                                           'Only work when --train False.')
    parser.add_argument('--psz', type=int, default='7',
                        help='Patch Size (default: 7)')
    config = parser.parse_args()
    main(config)
