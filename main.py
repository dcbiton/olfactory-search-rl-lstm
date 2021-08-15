from main_fcn_updown import learn_environment
import torch
import pickle

# mapping of action to index used only for checking
action_map = ["up", "down", "left", "right"]

# number of actions
num_actions = 4 
# epsilon value for epsion greedy
epsilon = 0.5
# gamma value for the learning rate
gamma = 1
# learning rate for the LSTM
sgd_learning_rate = 0.001
# sgd_learning_rate = 0.1
# momentum for the gradient descent
sgd_momentum = 0.1
# # define the number of eposides
# max_episodes = 200000

# initialize a seed
# np.random.seed(1234)
# torch.manual_seed(0)

run_num = 0
# hidden cell size for the LSTM
hidden_size=12
# length of the corridor
N = 5
# define the number of eposides
max_episodes = 1500000
# decay rate of epsilon
epsilon_decay = 0.000001
# probability to receive a correct signal
p_correct = 0.8
# # JUST TRAIN IT!
print("Running with: [N]", N, "[SGDLR]", sgd_learning_rate, "[SGDM]", sgd_momentum, "[HIDDEN SIZE]", hidden_size, "[MAX EP]", max_episodes, "[EPSILON DECAY]", epsilon_decay)
agent, t_maze, loss_list, episode_length_list, all_episode_reward, all_episode_cols, all_episode_cols_wrong =\
learn_environment(
    N=N+1,
    num_actions=num_actions,
    epsilon=epsilon,
    gamma=gamma,
    sgd_learning_rate=sgd_learning_rate,
    sgd_momentum=sgd_momentum,
    max_episodes=max_episodes,
    run_num=run_num,
    hidden_size=hidden_size, 
    folder_name="UpDown/hidden_size=%d/p_correct=%.3f/N=%d/NOTHING"%(hidden_size, p_correct, N),
    epsilon_decay=epsilon_decay,
    p_correct=p_correct)

# from main_fcn_4_retrain import learn_environment
# agent_file_name="files/collection_of_trained_models/episode_length_per_time_N=6_eps=0.500_gam=0.970_sgdlr=0.000_sgdmometum=0.100_maxep=100000_run=0/Agent_N=6_eps=0.500_gam=0.970_sgdlr=0.000_sgdmometum=0.100_maxep=100000_run=0.pth"
# agent, t_maze, loss_list, episode_length_list, all_episode_reward, all_episode_cols, all_episode_cols_wrong =\
# learn_environment(
#     N=N+1,
#     num_actions=num_actions,
#     epsilon=epsilon,
#     gamma=gamma,
#     sgd_learning_rate=sgd_learning_rate,
#     sgd_momentum=sgd_momentum,
#     max_episodes=max_episodes,
#     run_num=run_num,
#     hidden_size=hidden_size, 
#     folder_name="reruns/1",
#     epsilon_decay=epsilon_decay,
#     agent_file_name=agent_file_name)

# hidden_size=12
# # length of the corridor
# N = 10
# # define the number of eposides
# max_episodes = 300000
# # JUST TRAIN IT!
# agent, t_maze, loss_list, episode_length_list, all_episode_reward, all_episode_cols, all_episode_cols_wrong =\
# learn_environment(
#     N=N+1,
#     num_actions=num_actions,
#     epsilon=epsilon,
#     gamma=gamma,
#     sgd_learning_rate=sgd_learning_rate,
#     sgd_momentum=sgd_momentum,
#     max_episodes=max_episodes,
#     run_num=run_num,
#     hidden_size=hidden_size, 
#     folder_name="hidden_size=%d"%hidden_size)

# hidden_size=12
# # length of the corridor
# N = 20
# # define the number of eposides
# max_episodes = 300000
# # JUST TRAIN IT!
# agent, t_maze, loss_list, episode_length_list, all_episode_reward, all_episode_cols, all_episode_cols_wrong =\
# learn_environment(
#     N=N+1,
#     num_actions=num_actions,
#     epsilon=epsilon,
#     gamma=gamma,
#     sgd_learning_rate=sgd_learning_rate,
#     sgd_momentum=sgd_momentum,
#     max_episodes=max_episodes,
#     run_num=run_num,
#     hidden_size=hidden_size, 
#     folder_name="hidden_size=%d"%hidden_size)

# hidden_size=12
# # length of the corridor
# N = 30
# # define the number of eposides
# max_episodes = 300000
# # JUST TRAIN IT!
# agent, t_maze, loss_list, episode_length_list, all_episode_reward, all_episode_cols, all_episode_cols_wrong =\
# learn_environment(
#     N=N+1,
#     num_actions=num_actions,
#     epsilon=epsilon,
#     gamma=gamma,
#     sgd_learning_rate=sgd_learning_rate,
#     sgd_momentum=sgd_momentum,
#     max_episodes=max_episodes,
#     run_num=run_num,
#     hidden_size=hidden_size, 
#     folder_name="hidden_size=%d"%hidden_size)

# hidden_size=12
# # length of the corridor
# N = 50
# # define the number of eposides
# max_episodes = 300000
# # JUST TRAIN IT!
# agent, t_maze, loss_list, episode_length_list, all_episode_reward, all_episode_cols, all_episode_cols_wrong =\
# learn_environment(
#     N=N+1,
#     num_actions=num_actions,
#     epsilon=epsilon,
#     gamma=gamma,
#     sgd_learning_rate=sgd_learning_rate,
#     sgd_momentum=sgd_momentum,
#     max_episodes=max_episodes,
#     run_num=run_num,
#     hidden_size=hidden_size, 
#     folder_name="hidden_size=%d"%hidden_size)