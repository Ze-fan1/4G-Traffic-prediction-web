import numpy as np

# --- 基础评估指标 ---

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(true - pred))

def MSE(pred, true):
    return np.mean((true - pred) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))

def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))

# --- 自定义高值准确率指标 ---

def calc_custom_acc(preds, trues):
    """
    计算高值子集准确率
    preds/trues shape: (samples, pred_len, cells)
    """
    # 将时间维度展平，形状变为 (总预测时点数, 小区数)
    preds_flat = preds.reshape(-1, preds.shape[-1])
    trues_flat = trues.reshape(-1, trues.shape[-1])
    
    num_cells = trues_flat.shape[1]
    acc_list = []
    
    for i in range(num_cells):
        y = trues_flat[:, i]
        y_hat = preds_flat[:, i]
        
        # 1. 计算全量样本均值
        y_mean_all = np.mean(y)
        
        # 2. 筛选高于均值的子集 S
        S_mask = y >= y_mean_all
        if not np.any(S_mask):
            acc_list.append(0.0)
            continue
            
        y_S = y[S_mask]
        y_hat_S = y_hat[S_mask]
        
        # 3. 计算高值子集均值
        y_mean_S = np.mean(y_S)
        if y_mean_S == 0:
            acc_list.append(0.0)
            continue
            
        # 4. 计算子集上的 MAE
        MAE_S = np.mean(np.abs(y_S - y_hat_S))
        
        # 5. 结合截断条款计算最终 ACC
        raw_acc = np.abs(y_mean_S - MAE_S) / y_mean_S
        # 确保单小区结果稳健地落在 [0, 1] 之间
        acc_list.append(max(0.0, min(1.0, raw_acc)))
            
    return np.mean(acc_list)

# --- 统一汇总接口 ---

def metric(pred, true):
    """
    一键计算所有标准指标及自定义准确率
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    avg_acc = calc_custom_acc(pred, true)

    return mae, mse, rmse, mape, mspe, avg_acc