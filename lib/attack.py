from pathlib import Path
# pytorch
import torch
import torch.nn as nn
from torchvision.utils import save_image
# NVIDIA apex
from apex import amp
# math and showcase
import matplotlib.pyplot as plt
import numpy as np

from lib.utils import input_normalize
from lib.instructor import Instructor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_trans(grid_hist, real_hist, grad_hist, prefix, image_size=224, epsilon=1):
    image_dir = Path('./picture') / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    grid_trans = image_dir / 'grids.png'
    grad_trans = image_dir / 'grads.png'
    if len(grid_hist) < 1:
        print('Iteration smaller than 1, cannot show the grid transistion')
        return
    print('==> Show the grid transistion')
    value_range = (2.0/image_size)*epsilon
    fig, axs = plt.subplots(4, len(grid_hist), figsize=(6*len(grid_hist)+1, 24))
    for i, grid in enumerate(grid_hist):
        axs[0, i].imshow(grid[0,0,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
        axs[1, i].imshow(grid[0,1,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
    for i, grid in enumerate(real_hist):
        axs[2, i].imshow(grid[0,0,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
        axs[3, i].imshow(grid[0,1,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
    plt.savefig(str(grid_trans))
    plt.close()
    print('==> Show the gradient magnitude')
    x = range(1, len(grad_hist)+1)
    plt.xticks(range(1, len(grad_hist)+1, 1))
    plt.plot(x, grad_hist, label='gradient magnitude')
    plt.savefig(str(grad_trans))
    plt.close()

def show_example(data, adv, prefix):
    image_dir = Path('./picture') / prefix
    image_dir.mkdir(parents=True, exist_ok=True)
    normal_path = image_dir / 'example_normal.png'
    adv_path = image_dir / 'example_adversarial.png'
    # use the torchvision save_image method
    save_image(data[0:4], str(normal_path))
    save_image(adv[0:4], str(adv_path))

# adversarial attack
def baseline_1_1(model, instructor, optimizer, data, label, epsilon, step_size, iter, record=False):
    '''
    Spatial adversarial attack baseline 1.1
        Inside instructor:
        primitive grid -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget)
    '''
    grid_hist = [] # the original grid
    grad_hist = [] # gradient magnitude over iteration
    real_hist = [] # the final grid that bind with image

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
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            real_hist.append(grid.clone().detach().cpu().numpy())
    # bind it back to image
    samp_grid = instructor.prim_grid_2_samp_grid(grid)
    adv = instructor.image_binding(data, samp_grid)
    if record:
        show_trans(grid_hist, real_hist, grad_hist, 'baseline_1_1')
        show_example(data, adv, 'baseline_1_1')
    return adv

def baseline_2_1(model, instructor, optimizer, data, label, epsilon, kernel_size, step_size, iter, record=False):
    '''
    Spatial adversarial attack baseline 2.1
        add gaussian kernel for more smooth image outcome
        Inside instructor:
        primitive grid -> gaussian blur -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget)
    '''
    grid_hist = [] # the original grid
    grad_hist = [] # gradient magnitude over iteration
    real_hist = [] # the final grid that bind with image

    data = data.clone().detach().to(device)
    grid = instructor.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # instructor
        prim_grid = grid
        prim_grid = instructor.gaussian_blur(prim_grid, kernel_size)
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
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = instructor.gaussian_blur(grid, kernel_size)
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    prim_grid = instructor.gaussian_blur(grid, kernel_size)
    samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
    adv = instructor.image_binding(data, samp_grid)
    if record:
        show_trans(grid_hist, real_hist, grad_hist, 'baseline_2_1')
        show_example(data, adv, 'baseline_2_1')
    return adv

def baseline_2_2(model, instructor, optimizer, data, label, epsilon, kernel_size, step_size, iter, record=False):
    '''
    Spatial adversarial attack baseline 2.2
        add gaussian kernel for more smooth image outcome
        Inside instructor:
        primitive grid -> gaussian blur -> primitive grid clipping -> sampling grid -> image binding
        Outside:
        primitive grid clipping (for attack budget)
    '''
    grid_hist = [] # the original grid
    grad_hist = [] # gradient magnitude over iteration
    real_hist = [] # the final grid that bind with image

    data = data.clone().detach().to(device)
    grid = instructor.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # instructor
        prim_grid = grid
        prim_grid = instructor.gaussian_blur(prim_grid, kernel_size)
        prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
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
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = instructor.gaussian_blur(grid, kernel_size)
            prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    prim_grid = instructor.gaussian_blur(grid, kernel_size)
    prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
    samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
    adv = instructor.image_binding(data, samp_grid)
    if record:
        show_trans(grid_hist, real_hist, grad_hist, 'baseline_2_2')
        show_example(data, adv, 'baseline_2_2')
    return adv

def baseline_2_3(model, instructor, optimizer, data, label, epsilon, kernel_size, step_size, iter, record=False):
    '''
    Spatial adversarial attack baseline 2.3
        add gaussian kernel for more smooth image outcome
        Inside instructor:
        primitive grid -> gaussian blur -> primitive grid clipping -> sampling grid -> image binding
    '''
    grid_hist = [] # the original grid
    grad_hist = [] # gradient magnitude over iteration
    real_hist = [] # the final grid that bind with image

    data = data.clone().detach().to(device)
    grid = instructor.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # instructor
        prim_grid = grid
        prim_grid = instructor.gaussian_blur(prim_grid, kernel_size)
        prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
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
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            grad_hist.append(grad.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = instructor.gaussian_blur(grid, kernel_size)
            prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    prim_grid = instructor.gaussian_blur(grid, kernel_size)
    prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
    samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
    adv = instructor.image_binding(data, samp_grid)
    if record:
        show_trans(grid_hist, real_hist, grad_hist, 'baseline_2_3')
        show_example(data, adv, 'baseline_2_3')
    return adv

def baseline_3_3(model, instructor, optimizer, data, label, epsilon, kernel_size, step_size, iter, record=False):
    '''
    Spatial adversarial attack baseline 3.3
        add gaussian kernel for more smooth image outcome
        Inside instructor:
        primitive grid -> gaussian blur -> primitive grid clipping -> sampling grid -> sampling grid clipping -> image binding
    '''
    grid_hist = [] # the original grid
    grad_hist = [] # gradient magnitude over iteration
    real_hist = [] # the final grid that bind with image

    data = data.clone().detach().to(device)
    grid = instructor.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # instructor
        prim_grid = grid
        prim_grid = instructor.gaussian_blur(prim_grid, kernel_size)
        prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
        samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
        samp_grid = instructor.samp_clip(samp_grid, eta=1.0)
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
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            grad_hist.append(grad.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = instructor.gaussian_blur(grid, kernel_size)
            prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
            samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
            samp_grid = instructor.samp_clip(samp_grid, eta=1.0)
            prim_grid = instructor.samp_grid_2_prim_grid(samp_grid)
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    prim_grid = instructor.gaussian_blur(grid, kernel_size)
    prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
    samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
    samp_grid = instructor.samp_clip(samp_grid, eta=1.0)
    adv = instructor.image_binding(data, samp_grid)
    if record:
        show_trans(grid_hist, real_hist, grad_hist, 'baseline_3_3')
        show_example(data, adv, 'baseline_3_3')
    return adv

def baseline_4_3(model, instructor, optimizer, data, label, epsilon, kernel_size, step_size, iter, record=False):
    '''
    Spatial adversarial attack baseline 4.3
        add gaussian kernel for more smooth image outcome
        Inside instructor:
        primitive grid -> gaussian blur -> primitive grid clipping -> sampling grid -> sampling grid clipping -> image binding
    '''
    grid_hist = [] # the original grid
    grad_hist = [] # gradient magnitude over iteration
    real_hist = [] # the final grid that bind with image

    data = data.clone().detach().to(device)
    grid = instructor.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # instructor
        prim_grid = grid
        prim_grid = instructor.gaussian_blur(prim_grid, kernel_size)
        accu_grid = instructor.samp_grid_2_accu_grid(instructor.prim_grid_2_samp_grid(prim_grid))
        accu_grid = instructor.accu_clip(accu_grid)
        prim_grid = instructor.samp_grid_2_prim_grid(instructor.accu_grid_2_samp_grid(accu_grid))
        prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
        samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
        samp_grid = instructor.samp_clip(samp_grid, eta=1.0)
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
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            grad_hist.append(grad.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            prim_grid = instructor.gaussian_blur(grid, kernel_size)
            accu_grid = instructor.samp_grid_2_accu_grid(instructor.prim_grid_2_samp_grid(prim_grid))
            accu_grid = instructor.accu_clip(accu_grid)
            prim_grid = instructor.samp_grid_2_prim_grid(instructor.accu_grid_2_samp_grid(accu_grid))
            prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
            samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
            samp_grid = instructor.samp_clip(samp_grid, eta=1.0)
            prim_grid = instructor.samp_grid_2_prim_grid(samp_grid)
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    prim_grid = instructor.gaussian_blur(grid, kernel_size)
    accu_grid = instructor.samp_grid_2_accu_grid(instructor.prim_grid_2_samp_grid(prim_grid))
    accu_grid = instructor.accu_clip(accu_grid)
    prim_grid = instructor.samp_grid_2_prim_grid(instructor.accu_grid_2_samp_grid(accu_grid))
    prim_grid = instructor.prim_clip(prim_grid, rho=epsilon)
    samp_grid = instructor.prim_grid_2_samp_grid(prim_grid)
    samp_grid = instructor.samp_clip(samp_grid, eta=1.0)
    adv = instructor.image_binding(data, samp_grid)
    if record:
        show_trans(grid_hist, real_hist, grad_hist, 'baseline_4_3')
        show_example(data, adv, 'baseline_4_3')
    return adv

def baseline_5(model, instructor, optimizer, data, label, epsilon, kernel_size, step_size, iter, record=False):
    '''
    Spatial adversarial attack baseline 5
        add gaussian kernel for more smooth image outcome
        Inside instructor:
        primitive grid -> gaussian blur -> primitive grid clipping -> sampling grid -> sampling grid clipping -> image binding
    '''
    grid_hist = [] # the original grid
    grad_hist = [] # gradient magnitude over iteration
    real_hist = [] # the final grid that bind with image

    data = data.clone().detach().to(device)
    grid = instructor.init_prim_grid().detach().to(device)
    label = label.clone().detach().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(iter):
        grid.requires_grad = True
        # instructor
        prim_grid = grid
        samp_grid = instructor.forward_grid(prim_grid, rho=epsilon)
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
        # =========
        if record:
            grid_hist.append(grid.clone().detach().cpu().numpy())
            grad_hist.append(grad.clone().detach().cpu().numpy())
            # pass grid through the dataflow to get the real effect of grid
            # only differ while clipping / other operations inside gradient computation
            samp_grid = instructor.forward_grid(grid, rho=epsilon)
            prim_grid = instructor.samp_grid_2_prim_grid(samp_grid)
            real_hist.append(prim_grid.clone().detach().cpu().numpy())
    # bind it back to image
    samp_grid = instructor.forward_grid(grid, rho=epsilon)
    adv = instructor.image_binding(data, samp_grid)
    if record:
        show_trans(grid_hist, real_hist, grad_hist, 'baseline_5')
        show_example(data, adv, 'baseline_5')
    return adv