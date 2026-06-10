from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, calc_custom_acc, MAE, MSE, RMSE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # --- 闭环自回归 3 小时滚动预测逻辑 ---
                current_x = batch_x.clone()
                current_x_mark = batch_x_mark.clone()
                rolled_predictions = []

                for step_idx in range(8):
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([current_x[:, -self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if getattr(self.args, 'output_attention', False):
                                outputs = self.model(current_x, current_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(current_x, current_x_mark, dec_inp, batch_y_mark)
                    else:
                        if getattr(self.args, 'output_attention', False):
                            outputs = self.model(current_x, current_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(current_x, current_x_mark, dec_inp, batch_y_mark)

                    # 截断输入和输出的维度，防止潜在不一致
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                    # 1. 提取当前预测的前 3 小时
                    pred_3h = outputs[:, :3, :]
                    rolled_predictions.append(pred_3h)

                    # 2. 更新输入特征：去掉最老3小时，拼接新预测的3小时
                    current_x = torch.cat([current_x[:, 3:, :], pred_3h], dim=1)

                    # 3. 更新时间戳特征：时间窗口同样向后平移 3 小时
                    # 这里的 batch_y_mark 充当了未来时间轴的参考
                    start_mark_idx = self.args.label_len + step_idx * 3
                    next_mark_3h = batch_y_mark[:, start_mark_idx:start_mark_idx+3, :]
                    current_x_mark = torch.cat([current_x_mark[:, 3:, :], next_mark_3h], dim=1)

                # 将 8 次滚动的 3 小时预测拼接，形成最终该样本闭门造车的 24 小时预测值
                final_pred = torch.cat(rolled_predictions, dim=1) # [B, 24, D]

                # 获取真正的未来 24 小时真实标签进行对齐考核
                # batch_y 包含 [label_len + 24] 的长度，后 24 小时为 Ground Truth
                final_true = batch_y[:, self.args.label_len:self.args.label_len+self.args.pred_len, f_dim:]

                # 收集用于计算整体 Metric 的数据
                preds.append(final_pred.detach().cpu().numpy())
                trues.append(final_true.detach().cpu().numpy())

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = final_true.detach().cpu().numpy()
                    pd_val = final_pred.detach().cpu().numpy()
                    visual(gt[0, :, -1], pd_val[0, :, -1], os.path.join(folder_path, str(i) + '.pdf'))

# 使用 concatenate 可以完美合并最后批次(B)维度不一致的张量序列
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # concatenate 后已经是正确的三维形状 (总样本数, 24, 8)，无需再 reshape
        print('test shape:', preds.shape, trues.shape)

        # 结果保存
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw 计算（可选）
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        # --- 核心更新：MSE/MAE/RMSE在scaled空间，MAPE/MSPE/Custom_ACC在原始空间 ---
        mae = MAE(preds, trues)
        mse = MSE(preds, trues)
        rmse = RMSE(preds, trues)

        preds_inv = test_data.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
        trues_inv = test_data.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)

        trues_flat = trues_inv.reshape(-1)
        preds_flat = preds_inv.reshape(-1)
        mask = trues_flat > 1e-5
        mape = np.mean(np.abs((trues_flat[mask] - preds_flat[mask]) / trues_flat[mask])) if np.sum(mask) > 0 else 0.0
        mspe = np.mean(np.square((trues_flat[mask] - preds_flat[mask]) / trues_flat[mask])) if np.sum(mask) > 0 else 0.0
        avg_acc = calc_custom_acc(preds_inv, trues_inv)
        # --- 核心更新：打印所有返回的指标 ---
        print('MSE:{:.4f}, MAE:{:.4f}, RMSE:{:.4f}, MAPE:{:.4f}, MSPE:{:.4f}, DTW:{}, Custom_ACC:{:.4f}'.format(
            mse, mae, rmse, mape, mspe, dtw, avg_acc))
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('MSE:{:.4f}, MAE:{:.4f}, RMSE:{:.4f}, MAPE:{:.4f}, MSPE:{:.4f}, DTW:{}, Custom_ACC:{:.4f}\n'.format(
            mse, mae, rmse, mape, mspe, dtw, avg_acc))
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, avg_acc]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return