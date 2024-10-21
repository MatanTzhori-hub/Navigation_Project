import numpy as np
import os.path

import utils
    
def point_diff(start, end):
        x, y, theta = start
        x_e, y_e, theta_e = end
        
        x_f = x_e - x
        y_f = y_e - y
        theta_f = (theta_e - theta) % np.pi
        
        return x_f,y_f,theta_f
    

n_samples = 20
ratio = 0.2
titles = 'deltaX,deltaY,deltaTheta,Velocity,SteeringAngle,Time'
filespath = 'dataset'
train_file = 'overfitting_train.csv'
test_file = 'overfitting_test.csv'

max_delta = np.pi/4
min_v = 5
max_v = 10
min_T = 1
max_T = 1.5
L = 2
start = [0,0,0]

train_path = os.path.join(filespath, train_file)
test_path = os.path.join(filespath, test_file)

start = np.array(start)
dataset = np.empty(shape=(n_samples, 6))

v = np.random.uniform(min_v, max_v, size=n_samples)
delta = np.random.uniform(-max_delta, max_delta, size=n_samples)
T = np.random.uniform(min_T, max_T, size=n_samples)

for i in range(n_samples):
    goal = utils.destination(L, T[i], v[i], delta[i], start)
    goal = np.array(goal)
    
    diff = point_diff(start, goal)
    
    dataset[i] = np.concatenate((diff, [v[i], delta[i], T[i]]), axis=0)
    
    
test_size = (int)(np.floor(n_samples*ratio))
train_size = n_samples - test_size

train_ds = dataset[:train_size, :]
test_ds = dataset[train_size:, :]
np.savetxt(train_path, train_ds, delimiter=',', header=titles, comments='', fmt='%.6f')
np.savetxt(test_path, test_ds, delimiter=',', header=titles, comments='', fmt='%.6f')
    