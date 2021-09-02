import torch
import torch.nn as nn
from apex import amp

from lib.utils import input_normalize
from lib.instructor import Instructor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# adversarial attack
def baseline_1_1(model, instructor, optimizer, data, label, epsilon, step_size, iter=10):
    '''
    Spatial adversarial attack baseline 1.1
        Inside instructor:
        primitive grid -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget)
    '''
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
        grid = grid + step_size*sign_data_grad
        grid = grid.detach()
        # clippping
        prim_grid = grid 
        prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
        grid = prim_grid
        # =========
    # warp it back to image
    adv = instructor(data, grid, rho=epsilon, eta=1.0).detach()
    return adv