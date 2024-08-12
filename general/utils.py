'''   
Utils for saving results

'''


import os
import torch
import numpy as np
from datetime import datetime
from torchsummary import summary

def save_results(mean_pred, var_pred, model, save_dir, loss_history=None,loss_mse_history=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    date_str = datetime.now().strftime("%d%m_%H%M")
    mean_pred_file = os.path.join(save_dir, f'mean_predictions_{date_str}.npy')
    var_pred_file = os.path.join(save_dir, f'variance_predictions_{date_str}.npy')
    model_file = os.path.join(save_dir, f'model_{date_str}.pth')
    model_state_dict_file = os.path.join(save_dir, f'model_state_dict_{date_str}.pth')
    if loss_history is not None:
        loss_history_file = os.path.join(save_dir,f'loss_hist_{date_str}.npy')
        np.save(loss_history_file, loss_history.numpy())

    if loss_mse_history is not None:
        loss_mse_history_file = os.path.join(save_dir, f'loss_mse_hist_{date_str}.npy')    
        np.save(loss_mse_history_file, loss_mse_history.numpy())
        
    np.save(mean_pred_file, mean_pred.numpy())
    np.save(var_pred_file, var_pred.numpy())
    torch.save(model, model_file)
    torch.save(model.state_dict(), model_state_dict_file)

def print_model_details(model, input_size):
    summary(model, input_size=input_size)


