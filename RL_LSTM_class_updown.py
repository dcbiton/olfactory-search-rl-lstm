import torch
import numpy as np

# now we can create the class for the Q-learning using LSTM
class QLearning_LSTM:
    """
        Agent that uses LSTM to learn the Q value
        1. Receives observation that converts it to pytorch
        2. initialize memory
        3. initialize LSTM and the optimizer for the LSTM
        4. choose action using output from LSTM using epsilon greedy
        5. update the target using a formula
        6. update the weights of the LSTM by backpropagating
    """
    def __init__(self, num_actions, epsilon, gamma,
                observation,
                sgd_learning_rate=0.001, sgd_momentum=0.5, 
                hidden_size=12):
        """
            get initial observation from the environment
            and initialize LSTM and memory
            Input: 
                num_actions: int
                    number of actions that will be used for the output of LSTM
                epsilon: float
                    epsilon for epsilon greedy
                gamma: float
                    gamma for the target Q value
                observation: np array
                    observation that will be used for the input of LSTM
                sgd_learning_rate: float
                    learning rate for the SGD
                sgd_momentum: float
                    momentum for the SGD
            Updates:
                done: bool
                    initialize it to False
        """
        print("USING Class UPDOWN")
        # initialize some global parameters
        # number of actions needed
        self.num_actions = num_actions
        # value for epsilon greedy
        self.epsilon = epsilon
        # value for gamma
        self.gamma = gamma
        # save the LSTM learning rate into a global variable
        self.sgd_learning_rate = sgd_learning_rate
        # save also the momentum for the gradient descent into a global variable
        self.sgd_momentum = sgd_momentum

        # get initial observation from the environment as input
        self.y = self.receive_observation(observation)

        self.hidden_size = hidden_size
        # initialize the LSTM function that will learn
        # model contains the weights that will approximate the policy
        self.initialize_LSTM_model()
        # model expects that it is LSTM pytorch objects

        # create variable for updating if terminal
        self.done = False

    def receive_observation(self, observation):
        """
            receive the observation from the environment
            and convert to to the format needed by LSTM
            Input:
                observation: np array
                    observation for 1 dataset
            Output:
                pytorch tensor that is the shape needed by the LSTM
        """
        # reshape observation to size needed by the model
        observations_array = np.reshape(
            observation,
            newshape=(1, 1, np.shape(observation)[0]))
        # convert everything to pytorch
        return torch.from_numpy(observations_array).type(torch.float64)

    def get_initial_memory(self, observation_size, num_layers, ht_size, ct_size):
        """
            create the initial memory ht and ct
            Input: 
                observation size: int
                    size of the input of LSTM
                num_layers: int
                    number of LSTM layers
                ht_size: int
                    size of LSTM that will be used as output
                ct_size: int
                    size of hidden layers
            Output:
                ht: pytorch tensor
                    will be used for the hidden state
                ct: pytorch tensor
                    tensor for initial cell state
        """
        # num_layers: number of memory cell
        # observation_size 1 only since we input one at a time
        # ht_size: output size
        # ct_size: hidden unit size
        ht = torch.zeros(size=(num_layers, observation_size, ht_size)).type(torch.float64)
        ct = torch.zeros(size=(num_layers, observation_size, ct_size)).type(torch.float64)
        return ht, ct

    def initialize_LSTM_model(self):
        """
            Initialize LSTM weights, memory, and optimizer
        """
        # initialize LSTM
        # input_size 3 features
        # hidden_size=4 number of features in the hidden state
        # num_layers= 3 number of memory cells
        # proj_size number of expected output for each action 
        self.model = torch.nn.LSTM(
            input_size=3,
            hidden_size=self.hidden_size, 
            num_layers=1,
            proj_size=4)
        self.model.double()
    
        # initialize the memory
        self.initialize_memory()

        # initialize optimizer
        self.initialize_optimizer()

    def initialize_optimizer(self):
        # initialize the optimizer
        self.criterion = torch.nn.MSELoss(reduction="sum") 
        # self.criterion = torch.sum((output - target)**2) 
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.sgd_learning_rate,
            momentum=self.sgd_momentum)
        
        # self.optimizer = torch.optim.RMSprop(
        #     self.model.parameters(),
        #     lr=self.sgd_learning_rate)
        
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=self.sgd_learning_rate)
        
        # reset gradient
        self.optimizer.zero_grad()
    
    def initialize_memory(self):
        """
            initialize memory for LSTM
        """
        # intialize the memory for the LSTM
        self.ht, self.ct = self.get_initial_memory(
            observation_size=self.y.shape[0],
            num_layers=1,
            ht_size=self.num_actions,
            ct_size=self.hidden_size)
    
    def get_action_epsilon_greedy(self, q_tensor, eps):
        """
            choose an action using epsilon greedy
            Input:
                q_tensor: pytorch tensor
                    shape: shape from LSTM output
                eps: float
                    epsilon value in epsilon greedy
            Output:
                a: int
                    index of the action that will be taken 
        """
        # draw a random number for epsilon greedy
        if np.random.random()<eps:
            # choose a random action from all choices
            a = np.random.choice(np.arange(0, 4, 1))
        else:
            # get index of action with the mximum rewards
            # here make way for ties
            maximum_index = torch.where(q_tensor==torch.max(q_tensor))[-1]
            # choose a random action from all actions
            # with maximum estimated reward
            a = np.random.choice(maximum_index.numpy())
        return a

    # function for one step update 
    def choose_action_using_LSTM(self, y_input):
        """
            pass input (forward pass) to the LSTM
            and get action using epsilon greedy policy
            Input:
                y_input: pytorch tensor
                    shape: from LSTM input
                    input 
            Output:
                a: int
                    integer of the action that will be taken
        """
        # 1 initialize the LSTM function
        # get initial observation from the LSTM model
        self.q, (self.ht, self.ct) = self.model(y_input, (self.ht, self.ct))
        # pass self.q to the fully connected layer
        # print("[BEFORE SOFTMAX]", self.q)
        # self.q = self.q[:, -1, :]
        # self.q = torch.nn.functional.softmax(self.q, dim=2)
        # print("[AFTER]", self.q)
        # self.q = self.fc(self.q)
        # get actions using epsilon greedy
        self.a = self.get_action_epsilon_greedy(q_tensor=torch.clone(self.q), eps=self.epsilon)
        return self.a

    def update_target(self, reward, new_observation, is_done=False):
        """
            Returns the target that will be needed 
            based on the received reward and new observation
            after taking new action
            Input:
                reward: float
                    value of the reward
                new_observation: np array
                    new observation from format by the environment
            Output:
                target: float
                    target that will be used for backpropagation
        """
        if is_done:
          target = reward 
        else:
          # receive the new observation from the environment
          # and convert it to format needed by the LSTM
          y_prime = self.receive_observation(observation=new_observation)
          # use one forward pass to the LSTM without updating the memory
          q_prime, (__, __) = self.model(y_prime, (self.ht.detach(), self.ct.detach()))
          # q_prime = torch.nn.functional.softmax(q_prime, dim=2)
          # fix the target / frozen target by detaching 
          # to remove the gradient and treat the q prime as a scalar
          q_prime = q_prime.detach()
          # take action and update state
          target = reward + self.gamma*float(torch.max(q_prime))
        #   print("[R]", reward, "[Q]", float(torch.max(q_prime)), "[T]", target)
        return target

    # create a function for backpropagation
    def backpropagate(self, target_sequences, approximated_sequences):
        """
            Do the backpropagation
            Input:
                target_sequences: pytorch tensor
                    target sequences that will be used for backpropagation
                approximated_sequences: pytorch tensor
                    sequence of q value outputs from LSTM
            Output:
                value of the loss: float
        """
        # reset gradient
        self.optimizer.zero_grad()
        # define the criterion for the loss between two values
        loss = self.criterion(approximated_sequences, target_sequences)
        # change loss into square difference of scalar values
        # loss = torch.sum((approximated_sequences - target_sequences)**2)
        # gradient descent steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
        self.optimizer.step()
        # return value of the loss for record keeping 
        return loss.item()
