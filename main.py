from Solver_Minimize import *

# optimizer = TrajectoryOptimizer()
# initial_state = [0, 0, 0]
# goal_state = [0, 77, np.pi/2]

# fig = plt.figure()
# fig.suptitle(f'init={initial_state}, final={goal_state}', fontsize=16)
# plt.subplot(5,4,1)
# for j, v in enumerate([0, 0.5, 5, 20, 50]):
#     for i, t in enumerate(np.arange(0, np.pi/2, np.pi/8)):
#         plt.subplot(5,4,j*4+i+1)
        
#         initial_guess = [v, t]
#         v_optimal, phi_optimal = optimizer.solve(initial_state, goal_state, initial_guess)
#         optimizer.plot_trajectory(v_optimal, phi_optimal, initial_state, goal_state, initial_guess)


# plt.show()
    

optimizer = TrajectoryOptimizer()
path = [[0,0,0], [7,2,np.pi/2], [8,9,0], [15,11,np.pi/2], [16,16,0]]
initial_guess = [0,0]

fig = plt.figure()
# fig.suptitle(f'init={initial_state}, final={goal_state}', fontsize=16)
v_optimal, phi_optimal = initial_guess
initial_state = path[0]
for goal_state in path[1:]:
    v_optimal, phi_optimal = optimizer.solve(initial_state, goal_state, [v_optimal, phi_optimal])
    optimizer.plot_trajectory(v_optimal, phi_optimal, initial_state, goal_state, [v_optimal, phi_optimal])
    
    dest = optimizer.destination(v_optimal, phi_optimal, initial_state)
    plt.scatter(dest[0], dest[1], color='c', label='_Hidden label')
    initial_state = goal_state
    
plt.show()