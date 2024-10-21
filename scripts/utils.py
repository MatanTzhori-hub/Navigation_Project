import torch
import numpy as np

#optimal destination after T seconds in ackerman model
def destination(L, T, v, phi, initial_state):
    eps = 1e-5
    phi = phi + eps 
    x_s, y_s, theta_s = initial_state
    theta_f = theta_s + (v / L) * torch.tan(phi) * T
    x_f = L/torch.tan(phi)*(torch.sin(theta_f)-torch.sin(theta_s)) + x_s
    y_f = L/torch.tan(phi)*(-torch.cos(theta_f)+torch.cos(theta_s)) + y_s
    return x_f,y_f,theta_f