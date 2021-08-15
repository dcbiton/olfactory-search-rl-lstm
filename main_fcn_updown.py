import numpy as np
import torch
import matplotlib.pyplot as plt # show image for visualization
import pickle # save arrays for future use
from RL_LSTM_class_updown import QLearning_LSTM
from RL_LSTM_env_updown import TMazeEnv
from evaluate_model_fcn_updown import evaluate_model

import os


def create_folder(folder_name):
    try:
        os.makedirs(folder_name)    
        print("Directory " , folder_name ,  " Created ")
    except FileExistsError:
        print("Directory " , folder_name ,  " already exists")  


def learn_environment(
    N, num_actions, epsilon,
    gamma, sgd_learning_rate,
    sgd_momentum, max_episodes,
    hidden_size=12, run_num=0, folder_name=None,
    epsilon_decay=0.0001,
    p_correct=1):
    """
        Function for the RL-LSTM
        uses class TMazeEnv and QLearning_LSTM
        Input:
            N: int
                length of the corridor N-1
            num_actions: int
                number of actions available for the agent
            epsilon: float
                value for epsilon greedy
            gamma: float
                discount factor for target
            sgd_learning_rate: float
                learning rate for sgd
            sgd_momentum: float
                momentum value fo sgd
            max_episodes: int
                number of episodes
    """
    create_folder("files/"+folder_name)
    # initialize initial epsilon
    epsilon0 = 0.5
    # intialize the environment
    t_maze = TMazeEnv(N=N,probability_correct=p_correct)
    # get observation after setting the environment
    # initial observation from the initialization
    observation = t_maze.get_observation()
    # then give the observation to the agent
    # initialize the agent and give the initial observation
    agent = QLearning_LSTM(
        num_actions=num_actions,
        epsilon=epsilon,
        gamma=gamma,
        observation=observation,
        sgd_learning_rate=sgd_learning_rate,
        sgd_momentum=sgd_momentum,
        hidden_size=hidden_size)

    loss_list = []
    episode_length_list = []
    mean_loss = 100000
    all_episode_rows = []
    all_episode_cols = []
    all_episode_rows_wrong = []
    all_episode_cols_wrong = []
    all_episode_reward = []
    ave_corr = []
    ave_corr_episodes = []

    evaluated_episode_length = []
    evaluated_reward = []
    evaluated_loss = []
    evaluated_return = []

    num_of_correct = 0
    episode_num = 0
    percent_correct = 0
    sum_count = 0
    # loop through all episodes
    # for episode_num in range(max_episodes):
    # while episode_num < max_episodes:
    for episode_num in range(max_episodes):
        # create a container for the observations
        all_y = []
        all_target = []
        all_actions = []
        all_approx_q = []

        # initialize or reset the environment
        t_maze.initialize_environment(probability_correct=p_correct)
        # get observation after reseting
        observation = t_maze.get_observation()
        # but do not initialize the agent
        # initialize agent's memory
        agent.initialize_memory()
        # change the value of the epsilon\
        agent.epsilon = epsilon0/(1. + epsilon_decay*(episode_num)**1.05)
        epsilon_value = agent.epsilon
        # if (episode_num %1000 == 0): print('ep ', episode_num, )
        # initialize initial state of the environment
        # one episode
        i = 0
        is_done = False
        all_rows = []
        all_cols = []
        while not is_done:
            # agent will receive the new observation
            agent.y = agent.receive_observation(observation=observation)
            # Agent choose action based on observation using LSTM
            action = agent.choose_action_using_LSTM(y_input=agent.y)
            # action = set_of_actions[i]
            # Environment will use action
            # and return observations and reward
            reward, is_done = t_maze.take_action_update_state(action=action)
            all_rows.append(t_maze.row)
            all_cols.append(t_maze.col)
            # before updating observation
            # append needed information to the containers
            all_y.append(agent.y)
            all_actions.append(action)
            # get new observation that will be used for backpropagation later
            new_observation = t_maze.get_observation()
            # Agent will get reward from the environment
            # and update the target so it can backpropagate
            target = agent.update_target(
                reward=reward,
                new_observation=new_observation,
                is_done=is_done)
            if (episode_num == max_episodes-1):
              print('y {} a {} row {} col {} r {}'.format(agent.y, action, t_maze.row, t_maze.col, reward))
            # create target tensor by updating only the the action chosen
            # target_tensor = torch.clone(agent.q.detach())
            # # target_tensor = target_tensor.detach()
            # # use index from action and change the value to the target
            # target_tensor[0,0,action] = target
            # print(agent.q[0,0,action].shape)
            all_target.append(torch.tensor(target))
            all_approx_q.append(torch.clone(agent.q[0,0,action]))
            
            # get new observation
            observation = new_observation
            i += 1
            if i > 1*10**3:
                # print("[E]", episode_num,"[Q]", agent.q[0,0], "[R]",reward, "[loc]", t_maze.row, t_maze.col, "[T]", target, "[A]", action, "[EPS]", agent.epsilon, "FORCED")
                break
            
        if reward > 0:
            all_episode_rows.append(all_rows)
            all_episode_cols.append(all_cols)
        else:
            all_episode_rows_wrong.append(all_rows) 
            all_episode_cols_wrong.append(all_cols)

        # reshape y to make it one input tensor
        stacked_y = torch.stack(all_y)
        stacked_y = torch.reshape(stacked_y, shape=(stacked_y.shape[0], 1, stacked_y.shape[-1])).type(torch.float64)

        # stacked_q = torch.stack(all_approx_q)
        # stacked_q = torch.reshape(stacked_q, shape=(stacked_q.shape[0], stacked_q.shape[-1]))

        stacked_target = torch.stack(all_target).type(torch.float64)
        # stacked_target = torch.reshape(stacked_target, shape=(stacked_target.shape[0], stacked_target.shape[-1]))

        # reconstruct the q from all of the sequences
        q_all, __ = agent.model(stacked_y)
        # create one-hot encoding for all actions taken
        all_actions_np = np.zeros(shape=(len(all_actions), 1, 4))
        all_actions_np[np.arange(len(all_actions)), :, all_actions] = 1
        all_actions_tensor = torch.tensor(all_actions_np, dtype=bool)
        
        # filter all the chosen q using each action
        q_chosen = q_all[all_actions_tensor]
        # do the backpropagation
        loss = agent.backpropagate(
            target_sequences=stacked_target,
            approximated_sequences=q_chosen
        )
        loss_list.append(loss)
        episode_length_list.append(i)
        all_episode_reward.append(True if reward > 0 else False)

        if reward > 0:
            num_of_correct += 1
        sum_count +=1
        percent_correct = num_of_correct/sum_count
        # print episode 1000 
        if (episode_num==0): 
            print('ep ', episode_num, "MeanEpLength (over 1000)", i, "MeanCorrect (over 1000)", reward, "MeanLoss", loss, "eps", epsilon_value)

        if ((episode_num %1000 == 0)and (episode_num>0)): 

            # after the whole process, we evaluate the model using epsilon=0 
            # and obtain the average episode length, reward, loss and return
            evaluation = evaluate_model(
                agent=agent,
                N=N,
                p_correct=p_correct,
                max_episodes=10**2,
                epsilon=0
            )

            # get the mean values of the arrays evaluated 
            mean_episode_length = np.mean(evaluation[0])
            mean_reward = np.mean(evaluation[1])
            mean_loss = np.mean(evaluation[2])
            mean_return  = np.mean(evaluation[3])

            # then store it in the list
            evaluated_episode_length.append(mean_episode_length)
            evaluated_reward.append(mean_reward)
            evaluated_loss.append(mean_loss)
            evaluated_return.append(mean_return)

            ave_corr.append(percent_correct)
            ave_corr_episodes.append(episode_num)
            if (episode_num %1000 == 0):
                print('ep ', episode_num, "MeanEpLength (over 1000)", mean_episode_length, "MeanCorrect (over 1000)", mean_reward, "MeanLoss", mean_loss, "MeanReturn", mean_return, "eps", epsilon_value)
            num_of_correct = 0
            sum_count = 0
            # save trained model for future use
        if ((episode_num %10000 == 0)and (episode_num>0)): 
            torch.save(agent, "files/"+folder_name+"/Agent_SUBEP_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_epnum=%d_run=%d.pth" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, episode_num, run_num))

            # save trained model for future use
            torch.save(t_maze, "files/"+folder_name+"/TMaze_SUBEP_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_epnum=%d_run=%d.pth" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, episode_num, run_num))

        # stop the run when the loss is too low
        if np.abs(mean_loss) < 0.02:
            print("STOPPING AT", episode_num,"MeanEpLength (over 1000)", mean_episode_length, "MeanCorrect (over 1000)", mean_loss, "MeanLoss", loss, "MeanReturn", mean_return, "[EPS]", epsilon_value, "END")
            break

    print("[EPISODE NUM]", episode_num, "[LOSS]", loss,"[FINAL REWARD]", reward, "[EPISODE LENGTH]", i)

    # convert to array for easy manipulation
    all_episode_reward = np.array(all_episode_reward, dtype=bool)
    loss_list = np.array(loss_list)
    episode_length_list = np.array(episode_length_list)


    # save trained model for future use
    torch.save(agent, "files/"+folder_name+"/Agent_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pth" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num))

    # save trained model for future use
    torch.save(t_maze, "files/"+folder_name+"/TMaze_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pth" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num))

    # save the loss array into a pickle file for future use
    with open("files/"+folder_name+"/loss_N=%d_epsdecay=%f_gam=%f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(loss_list, f)
    with open("files/"+folder_name+"/episode_length_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(episode_length_list, f)
    with open("files/"+folder_name+"/all_episode_reward_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(all_episode_reward, f)
    with open("files/"+folder_name+"/all_episode_cols_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(all_episode_cols, f)
    with open("files/"+folder_name+"/all_episode_cols_wrong_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(all_episode_cols_wrong, f)
    with open("files/"+folder_name+"/ave_corr_episodes_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(ave_corr_episodes, f)
    with open("files/"+folder_name+"/ave_corr_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(ave_corr, f)
    with open("files/"+folder_name+"/evaluated_episode_length_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(evaluated_episode_length, f)
    with open("files/"+folder_name+"/evaluated_reward_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(evaluated_reward, f)
    with open("files/"+folder_name+"/evaluated_loss_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(evaluated_loss, f)
    with open("files/"+folder_name+"/evaluated_return_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.pkl" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), 'wb') as f:
        pickle.dump(evaluated_return, f)

    # save the image for visualization
    plt.figure()
    correct_loss_episodes = loss_list[np.where(all_episode_reward==True)]
    correct_episode_numbers = np.arange(0, len(loss_list), 1)[np.where(all_episode_reward==True)]
    wrong_loss_episodes = loss_list[np.where(all_episode_reward==False)]
    wrong_episode_numbers = np.arange(0, len(loss_list), 1)[np.where(all_episode_reward==False)]
    plt.plot(np.arange(0, len(loss_list), 1), loss_list, "--", color="gray")
    plt.plot(correct_episode_numbers, correct_loss_episodes, "o", color="C0", label="correct")
    plt.plot(wrong_episode_numbers, wrong_loss_episodes, "o", color="C1", label="wrong")
    plt.ylabel("Loss")
    plt.xlabel("episodes")
    plt.legend()
    plt.savefig("files/"+folder_name+"/loss_per_time_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.png" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), dpi=300)
    plt.show()
    plt.close()

    plt.figure()
    correct_length_episodes = episode_length_list[np.where(all_episode_reward)]
    correct_length_numbers = np.arange(0, len(episode_length_list), 1)[np.where(all_episode_reward)]
    wrong_length_episodes = episode_length_list[np.where(~all_episode_reward)]
    wrong_length_numbers = np.arange(0, len(episode_length_list), 1)[np.where(~all_episode_reward)]
    plt.plot(np.arange(0, len(episode_length_list), 1), episode_length_list, "--", color="gray")
    plt.plot(correct_length_numbers, correct_length_episodes, "o", color="C0", label="correct")
    plt.plot(wrong_length_numbers, wrong_length_episodes, "o", color="C1", label="wrong")
    plt.ylim(0, 10+np.max(np.hstack((correct_length_episodes, wrong_length_episodes))))
    plt.ylabel("Episode Length")
    plt.xlabel("episodes")
    plt.legend()
    plt.savefig("files/"+folder_name+"/episode_length_per_time_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.png" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), dpi=300)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(ave_corr_episodes, ave_corr, "--", color="gray")
    plt.ylabel("Average Percent Correct")
    plt.xlabel("episodes")
    plt.savefig("files/"+folder_name+"/ave_corr_episodes_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.png" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), dpi=300)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(evaluated_episode_length, "--", color="C0")
    plt.ylabel("Episode Length")
    plt.xlabel("episodes")
    plt.savefig("files/"+folder_name+"/evaluated_episode_length_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.png" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), dpi=300)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(evaluated_reward, "--", color="C0")
    plt.ylabel("Average Reward")
    plt.xlabel("episodes")
    plt.savefig("files/"+folder_name+"/evaluated_reward_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.png" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), dpi=300)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(evaluated_loss, "--", color="C0")
    plt.ylabel("Average Loss")
    plt.xlabel("episodes")
    plt.savefig("files/"+folder_name+"/evaluated_loss_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.png" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), dpi=300)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(evaluated_return, "--", color="C0")
    plt.ylabel("Average Return")
    plt.xlabel("episodes")
    plt.savefig("files/"+folder_name+"/evaluated_return_N=%d_epsdecay=%f_gam=%.3f_sgdlr=%f_sgdmometum=%.3f_maxep=%d_run=%d.png" %(N, epsilon_decay, gamma, sgd_learning_rate, sgd_momentum, max_episodes, run_num), dpi=300)
    plt.show()
    plt.close()

    return agent, t_maze, loss_list, episode_length_list, all_episode_reward, all_episode_cols, all_episode_cols_wrong
