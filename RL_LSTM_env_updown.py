import numpy as np

class TMazeEnv:
    """
        T-maze environment
        Contains 3 things:
        1. intialize the environment of where the reward should be
        2. get observation based on the current state (stored in the class)
        3. take action and update the state
    """
    def __init__(self, N, probability_correct, set_reward=False, reward_state=0):
        # print("USING ENV UPDOWN REVISED")
        # limit of the environment
        # length of the corridor
        self.N = N

        # initialize the state of the environment
        # it will populate values to self.final_reward_state
        # and gupdate the initial value of signal
        self.initialize_environment(set_reward=set_reward,
                                    reward_state=reward_state,
                                    probability_correct=probability_correct)

    # first, create a function that will initialize the environment
    # meaning, it will initialize the initial signal that it will receive 
    # and initialize where the reward will be
    def initialize_environment(self, probability_correct, set_reward=False, reward_state=0):
        """
            Initialize the environment
            1. intialize the done variable
            2. initialize the state of where the final reward will be 
                and the initial observation
            3. intialize the state of the agent
            Return: 
                None
        """
        # initialize a variable that indicates if the episode is done
        self.done=False
        # set the probability of receiving the signal
        self.probability_correct = probability_correct
        # set the environment state
        if set_reward == True:
            self.final_reward_state = reward_state
        else:
            # choose where to put the reward, either up or down
            self.final_reward_state = np.random.randint(0, 2)
        # self.final_reward_state = 1
        # update the observation that will be given to the learner
        # observation will be numpy array
        if self.final_reward_state==0: # reward is up
            self.observation = np.array([1, 1, 0]) # up
        else: # reward is down
            self.observation = np.array([0, 1, 1]) # down
        # the observation will be used in the corridor
        # initialize state of the agent? 
        # we can think of it as the environment can see the state of the agent
        # and return the observation to the agent
        # but the agent shouldn't know its real state (y)
        self.row = 1
        self.col = 0

    # get observations for states that needs observation
    # i think i want a function that returns the current observation
    def get_observation(self):
        """
            Returns the current observation based on the current state (row and col)
            Does not change anything in the state
            Return:
                observation: np array
        """
        # get observation based on row and col
        # use row as the row index and col as the column index
        if self.row == 1:
            if self.col < self.N:
                random_number = np.random.random()
                if random_number<self.probability_correct:
                    if self.final_reward_state==0: # reward is up correct signal
                        self.observation = np.array([1, 1, 0]) # up
                    else: # reward is down
                        self.observation = np.array([0, 1, 1]) # down
                else:
                    if self.final_reward_state==0: # reward is up wrong signal
                        self.observation = np.array([0, 1, 1]) # down
                    else: # reward is down wrong signal
                        self.observation = np.array([1, 1, 0]) # up
            elif self.col == self.N:
                # at N, the agent is in the junction
                self.observation = np.array([0,1,0])
        return self.observation

    # take action given by external agent
    # this function is the comunication between environment and agent
    # every time you take action, you get a reward
    # the reward will not be given in any other way
    def take_action_update_state(self, action):
        """
            Use the input action to change the state
            Returns the reward based on the state
            Input:
                action: int in [0,1,2,3]
                    0 action up
                    1 action down
                    2 action left
                    3 action right
                    else: nothing will happen
            Return:
                reward: float
                    reward based on the update on the state
                done: bool
                    flag that tells if the agent is in the terminal state
        """
        # row = 1 col < N corridor
        # row = 1 col = 0 initial state
        # row = 1 col = N decision point
        # row = 0 col = N up
        # row = 2 col = N down

        # action 0: up
        if action == 0:
            new_row = self.row - 1
            new_col = self.col
        # action 1: down
        elif action == 1:
            new_row = self.row + 1
            new_col = self.col
        # action 2: left
        elif action == 2:
            new_row = self.row
            new_col = self.col - 1
        # action 3: right
        elif action == 3:
            new_row = self.row
            new_col = self.col + 1

        # get reward and update the state
        reward = -0.1
        if new_col == self.N:
            if new_row == 0: # up
                # if self.final_reward_state == 0, the reward is up
                reward = 4 if not bool(self.final_reward_state) else -0.1
                self.done = True
            elif new_row == 2: # down
                # if self.final_reward_state == 1, the reward is down
                reward = 4 if bool(self.final_reward_state) else -0.1
                self.done = True
        elif new_col > self.N:  # right 
            # if it is trying to move left, don't move
            reward = -0.1
            new_col = self.col
        elif new_row != 1: # do not move 
            new_row = self.row
            reward = -0.1 # negative reward
        elif new_col < 1:
            new_row = self.row
            new_col = self.col
            reward = -0.1 # negative reward
        elif new_row < 0:# do not move 
            new_row = self.row
            new_col = self.col
            reward = -0.1 # negative reward

        # update the state: rows and cols
        self.row = new_row
        self.col = new_col
        return reward, self.done
