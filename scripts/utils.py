import torch
import numpy as np

#optimal destination after T seconds in ackerman model
def destination(L, T, v, phi, initial_state):
    eps = 1e-5
    phi = phi + eps 
    x_s, y_s, theta_s = initial_state
    theta_f = theta_s + (v / L) * np.tan(phi) * T
    x_f = L/np.tan(phi)*(np.sin(theta_f)-np.sin(theta_s)) + x_s
    y_f = L/np.tan(phi)*(-np.cos(theta_f)+np.cos(theta_s)) + y_s
    return x_f,y_f,theta_f