from pathlib import Path
# pytorch
import torch
import torch.nn as nn
# NVIDIA apex
from apex import amp
# math and showcase
import matplotlib.pyplot as plt
import numpy as np

from lib.utils import input_normalize
from lib.instructor import Instructor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_trans(grid_hist, grad_hist, prefix, image_size=224, epsilon=1):
    image_dir = Path('./picture')
    grid_trans = image_dir / (prefix + '_grids.png')
    grad_trans = image_dir / (prefix + '_grads.png')
    if len(grid_hist) < 1:
        print('Iteration smaller than 1, cannot show the grid transistion')
        return
    print('==> Show the grid transistion')
    value_range = (2.0/image_size)*epsilon
    fig, axs = plt.subplots(4, len(grid_hist), figsize=(6*len(grid_hist)+1, 24))
    for i, grid in enumerate(grid_hist):
        axs[0, i].imshow(grid[0,0,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
        axs[1, i].imshow(grid[0,1,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
    plt.savefig(str(grid_trans))
    plt.close()
    print('==> Show the gradient magnitude')
    plt.plot(range(len(grad_hist)), grad_hist, label='gradient magnitude')
    plt.savefig(str(grad_trans))
    plt.close()

# adversarial attack
def baseline_1_1(model, instructor, optimizer, data, label, epsilon, step_size, iter, record=False):
    '''
    Spatial adversarial attack baseline 1.1
        Inside instructor:
        primitive grid -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget)
    '''
    grid_hist = []
    grad_hist = []

    data = data.clone().detach().to(device)
    grid = instructor.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # instructor
        prim_grid = grid
        samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
        adv = instructor.image_binding(data, samp_grid)
        # ==========
        output = model(input_normalize(adv))
        loss = criterion(output, label)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        sign_data_grad = grid.grad.sign()
        grad = torch.sum(torch.abs(grid.grad)[0].type(torch.float64))
        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        prim_grid = grid 
        prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
        grid = prim_grid
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            grad_hist.append(grad.clone().detach().cpu().numpy())
    # warp it back to image
    adv = instructor(data, grid, rho=epsilon, eta=1.0).detach()
    if record:
        show_trans(grid_hist, grad_hist, 'baseline_1_1')
    return adv