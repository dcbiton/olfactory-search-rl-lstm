import numpy as np
from RL_LSTM_env_updown import TMazeEnv
import torch
import matplotlib.pyplot as plt # show image for visualization
import pickle # save arrays for future use


def computeMSETorch(target_sequences, approximated_sequences):
    criterion = torch.nn.MSELoss(reduction="sum")
    loss = criterion(target_sequences, approximated_sequences)
    return loss


def evaluate_model(agent, N, p_correct, max_episodes, epsilon):
    # to evaluate the LSTM model, 
    # we need the agent
    # and we need to initialize the t_maze

    # intialize Tmaze
    t_maze = TMazeEnv(N=N, probability_correct=p_correct)
    # change epsilon
    agent.epsilon = epsilon

    # save all the rewards for each episode 
    all_episode_reward = [] # bool
    # save all the episode lengths 
    episode_length_list = [] 
    # save all returns
    all_episode_returns = []
    # save all loss 
    all_episode_loss = []

    # loop through all episodes
    for episode_num in range(max_episodes):
        # store all target for computing loss later
        all_target = []
        # store all y input for computing loss later
        all_y = []
        # store all actions taken during the whole episode
        all_actions = []

        # initialize or reset the environment
        t_maze.initialize_environment(probability_correct=p_correct)
        # get the observation after resetting
        observation = t_maze.get_observation()
        # do not initialize the agent, 
        # only initialize the agent's memory
        agent.initialize_memory()
        # define is_done
        is_done = False

        # counter i
        i = 0
        # initial return
        g = 0
        # loop through the whole episode
        while not is_done:
            # agent will receive the new observation
            agent.y = agent.receive_observation(observation=observation)
            # agent choose action based on observation using LSTM
            action = agent.choose_action_using_LSTM(y_input=agent.y)
            # environment will use action
            # and return observations and reward
            reward, is_done = t_maze.take_action_update_state(action=action)
            # agent will get new observation that will return target
            new_observation = t_maze.get_observation()
            # update the target so it backpropagates
            target = agent.update_target(
                reward=reward,
                new_observation=new_observation,
                is_done=is_done
            )

            # then store the needed values into the list
            all_target.append(torch.tensor(target))
            all_y.append(agent.y)
            all_actions.append(action)

            # compute return
            g += (agent.gamma**i)*reward
            # get new observation after taking action
            observation = np.copy(new_observation)
            # increase counter
            i += 1

            # force stop the episodes
            if i > 10**3:
                break
        
        # after the whole episode, use the target and the q value
        # then we can compute the loss
        # first, reshape the tensors
        stacked_y = torch.stack(all_y)
        stacked_y = torch.reshape(stacked_y, shape=(stacked_y.shape[0], 1, stacked_y.shape[-1])).type(torch.float64)
        # then reshape the target
        stacked_target = torch.stack(all_target).type(torch.float64)
        # then reobtain q from all the sequences
        q_all, __ = agent.model(stacked_y)
        # create one-hot encoding for all actions taken
        all_actions_np = np.zeros(shape=(len(all_actions), 1, 4))
        all_actions_np[np.arange(len(all_actions)), :, all_actions] = 1
        all_actions_tensor = torch.tensor(all_actions_np, dtype=bool)
        # then filter all the chosen q using each action
        q_chosen = q_all[all_actions_tensor]

        # then compute the loss
        loss = computeMSETorch(
            target_sequences=stacked_target,
            approximated_sequences=q_chosen
        )


        # episode_length = i
        episode_length_list.append(i)
        # episode_reward = True if reward == 4 else False
        all_episode_reward.append(True if reward > 0 else False)
        # get LOSS
        all_episode_loss.append(loss.detach().numpy())
        # save average return
        all_episode_returns.append(g)
    
    return episode_length_list, all_episode_reward, all_episode_loss, all_episode_returns