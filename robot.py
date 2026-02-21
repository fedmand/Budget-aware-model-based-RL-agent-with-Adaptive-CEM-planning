####################################
#      YOU MAY EDIT THIS FILE      #
# ALL OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

# Imports from this project
# You should not import any other modules, including config.py
# If you want to create some configuration parameters for your algorithm, keep them within this robot.py file
import config
import constants
from graphics import VisualisationLine

# Configure matplotlib for interactive mode
plt.ion()

# CONFIGURATION PARAMETERS HERE. Add whatever configuration parameters you like here.
# Remember, you will only be submitting this robot.py file, no other files.
SEED = 4

# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self):
        # The environment (only available during development mode)
        self.environment = None
        # A list of visualisations which will be displayed on the bottom half of the window
        self.visualisation_lines = []
        ## config params
        self.bootstrap_num_batches = 150
        self.bootstrap_steps = 1000
        self.replan_every = 5
        self.retrain_every = 30
        self.retrain_num_batches = 12
        self.end_training_margin = 0.1
        # flags to track phase
        self.bootstrap_trained = False
        self.testing_phase = False
        # CEM params
        self.NUM_ITER = 3
        self.NUM_PATH = 70
        self.NUM_ELITES = 10  # should always be 10-15% of paths
        self.HORIZON = 15
        # robot components
        self.replay_buffer = ReplayBuffer()
        self.dynamics_model = DynamicsModel()   
        self.distance_predictor = DistancePredictor()
        # plan 
        self.planned_actions = []
        self.num_steps = 0
        self.plan_steps = 0
        # stuck logic
        self.best_dist = np.inf
        self.no_improve_counter = 0
        self.no_move_counter = 0
        self.pending_reset = False
        self.reset_eps = 0.02
        self.move_eps = 1e-3
        self.reset_patience_boot = 100
        self.reset_patience_plan = 50
        self.patience_move = 30
        # DEBUG ONLY
        self.next_print = constants.INIT_MONEY - 10

    # Get the next training action
    def training_action(self, obs, money):
        # DEBUG ONLY
        if money <= self.next_print:
            self.next_print -= 10
            print(money)
            
        # money basically finished --> end training to avoid penalty and reset robot's statefor testing
        if money <= self.end_training_margin:
            action_type = 4
            action_value = np.zeros(2, dtype=np.float32)
            self.testing_phase = True
            self.reset()
            
        # robot has been stuck for patience steps --> reset 
        elif self.pending_reset and money > constants.COST_PER_RESET:
            action_type = 2
            action_value = np.zeros(2, dtype=np.float32)
            self.reset()
            print("Reset due to stuck")
        
        # Bootstrap phase: random action for exploration 
        elif self.replay_buffer.size < self.bootstrap_steps:
            action_type = 1
            action_value = np.random.uniform(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE, 2)
        elif self.replay_buffer.size == self.bootstrap_steps and not self.bootstrap_trained:  
            self.bootstrap_trained = True
            # train dynamics model and distance predictor
            self.dynamics_model.train(self.replay_buffer, self.bootstrap_num_batches)
            self.distance_predictor.train(self.replay_buffer, self.bootstrap_num_batches)
            # reset the robot to exploit planning on effectively collected data
            action_type = 2
            action_value = np.zeros(2, dtype=np.float32)
            self.reset()  
            print("Bootstrap finished")
        
        # actually plan with CEM
        else:
            action_type = 1
            if self.num_steps == 0 or self.num_steps % self.replan_every == 0:
                self.make_CEM_plan(obs)
                self.plan_steps = 0
            
            action_value = self.planned_actions[self.plan_steps]    # out-of-bounds will never happen as long as plan_steps < HORIZON
            self.plan_steps += 1
            self.num_steps += 1
            
            if self.num_steps % self.retrain_every == 0:
                self.dynamics_model.train(self.replay_buffer, self.retrain_num_batches)
                self.distance_predictor.train(self.replay_buffer, self.retrain_num_batches)
                
        return action_type, action_value

    # Get the next testing action
    def testing_action(self, obs):
        if self.num_steps % self.replan_every == 0:
            self.make_CEM_plan(obs)
            self.plan_steps = 0
            
        action = self.planned_actions[self.plan_steps]
        self.plan_steps += 1
        self.num_steps += 1
        return action

    def receive_transition(self, obs, action, next_obs, distance_to_goal):
        # Receive a transition --> add it to replay buffer
        self.replay_buffer.add_transition(obs, action, next_obs, distance_to_goal)
        
        # goal has been reached --> reset because modelling after red line is wasteful
        if distance_to_goal <= 0.01:
            print(f"Goal reached within {distance_to_goal} --> reset")
            self.pending_reset = True
            return
        
        # if robot is stuck for patience steps --> raise reset flag (actual reset will be done in hte next training_action call) 
        if distance_to_goal < self.best_dist - self.reset_eps:
            self.best_dist = distance_to_goal
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1
        
        if self.replay_buffer.size < self.bootstrap_steps:
            patience = self.reset_patience_boot
        else:
            patience = self.reset_patience_plan
            
        # no-move counter: observation space proxy to detect stuck. Vertical distances are ok
        obs = np.asarray(obs)
        next_obs = np.asarray(next_obs)
        delta = np.linalg.norm(next_obs - obs)
        if delta <= self.move_eps:
            self.no_move_counter += 1
        else:
            self.no_move_counter = 0
        
        # reset <--> no distance improvement && no movement    
        if self.no_improve_counter >= patience and self.no_move_counter >= self.patience_move:
            self.pending_reset = True
        
        
    # Receive a new demonstration
    def receive_demo(self, demo):
        pass
    
    def reset(self):
        self.planned_actions = []
        self.num_steps = 0
        self.plan_steps = 0
        self.no_improve_counter = 0
        self.no_move_counter = 0
        self.best_dist = np.inf
        self.pending_reset = False
        self.goal_counter = 0
    
    # The CEM planning algorithm
    def make_CEM_plan(self, obs):
        H = self.HORIZON
        N = self.NUM_PATH
        I = self.NUM_ITER
        E = self.NUM_ELITES
        Amax = constants.MAX_ACTION_MAGNITUDE

        # Current search distribution
        mean = np.zeros((H, 2), dtype=np.float32)
        # variance scaled based on robot's learning phase
        if not self.bootstrap_trained:
            scale = 0.5
        elif self.bootstrap_trained and not self.testing_phase:
            scale = 0.25
        else:   
            scale = 0.1    # testing phase
        std  = np.ones((H, 2), dtype=np.float32) * (scale * Amax)  # to ease exploration

        obs0 = np.array(obs, dtype=np.float32)

        # Reuse buffer for sampled action sequences for this iteration only
        actions = np.empty((N, H, 2), dtype=np.float32)
        dists   = np.empty((N,), dtype=np.float32)

        for it in range(I):
            # Sample action sequences
            if it == 0:
                actions[:] = np.random.uniform(-Amax, Amax, size=(N, H, 2)).astype(np.float32)
            else:
                actions[:] = np.random.normal(loc=mean, scale=std, size=(N, H, 2)).astype(np.float32)
                np.clip(actions, -Amax, Amax, out=actions)

            # Batched rollout
            curr_obs = np.repeat(obs0[None, :], N, axis=0)  # (N,3)
            for t in range(H):
                curr_obs = self.dynamics_model.predict_next_obs_batch(curr_obs, actions[:, t, :])

            # Batched terminal distance: curr_obs is the last obs for each of the N paths 
            dists[:] = self.distance_predictor.predict_distance_batch(curr_obs).astype(np.float32)

            # Elite selection: partial (no full sort)
            elite_idx = np.argpartition(dists, E-1)[:E]
            elite_actions = actions[elite_idx]  # (E,H,2)

            # Update distribution
            mean = elite_actions.mean(axis=0)
            std  = elite_actions.std(axis=0) + 1e-6  # numerical floor

        self.planned_actions = mean.astype(np.float32)

        
        
        
class ReplayBuffer():
    def __init__(self):
        self.size = 0
        self.buffer = []
        
    def add_transition(self, obs, action, next_obs, distance_to_goal):
        transition = (
            np.array(obs, dtype=np.float32),  # to facilitate things later
            np.array(action, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            float(distance_to_goal)
            )
        self.buffer.append(transition)
        self.size += 1
        
    def sample_batch(self, batch_size):
        if batch_size > self.size:
            raise ValueError(f"batch size = {batch_size} > buffer size = {self.size}")
        
        # random batch_size indices from 0 to self-size with no replacement
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # unpack data structure: now obs contains batch_size nparrays of 3 elements each 
        obs, actions, next_obs, distances = zip(*batch)  # basically performs a matrix transpose operation
        # stack all nparrays row-wise which corresponds to axis=0 (default) to obtain (batch_size, feature_dim), i.e. I specify the dimension along which I am stacking (0=rows)
        return np.stack(obs), np.stack(actions), np.stack(next_obs), np.array(distances).reshape(-1, 1)
        
        
        
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class DynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # o_t+1 = NN(o_t, a_t) . Obs are 3D, actions 2D
        self.network = MLP(5, 64, 3)
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate) 
        self.loss = nn.MSELoss()
        
    def train(self, buffer, num_batch):
        self.network.train()
        for b in range(num_batch):
            # prepare data
            obs, actions, next_obs, distances = buffer.sample_batch(self.batch_size)
            inputs = np.concatenate([obs, actions], axis=1)
            inputs = torch.from_numpy(inputs).float() 
            targets = torch.from_numpy(next_obs).float()   

            # forward, backward, update
            predictions = self.network(inputs)
            loss = self.loss(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
         
            
    def predict_next_obs(self, obs, action):
        self.network.eval()
        with torch.no_grad():
            # format input and add batch dimension
            input = np.concatenate([obs, action], axis=-1)
            input = torch.from_numpy(input).float().unsqueeze(0)  # now (1,5)
            predicted_obs = self.network(input).squeeze(0).numpy()   # return (3,)
            return predicted_obs
        
    def predict_next_obs_batch(self, obs_batch, act_batch):
        # obs: (N,3), act: (N,2) -> next_obs: (N,3)
        self.network.eval()
        with torch.no_grad():
            x = np.concatenate([obs_batch, act_batch], axis=1)  # (N,5)
            x = torch.from_numpy(x).float()
            y = self.network(x)    #(N,3)
            return y.numpy().astype(np.float32)

         
class DistancePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # d_t = NN(o_t)
        self.network = MLP(3, 64, 1)
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate) 
        self.loss = nn.MSELoss()    
        
    def train(self, buffer, num_batch):
        self.network.train()
        for b in range(num_batch):
            # prepare data
            obs, actions, next_obs, distances = buffer.sample_batch(self.batch_size)
            inputs = torch.from_numpy(next_obs).float() 
            targets = torch.from_numpy(distances).float()  #add feature dimension   

            # forward, backward, update
            predictions = self.network(inputs)
            loss = self.loss(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()    
        
    def predict_distance(self, obs):
        """
        Requires obs to be a Numpy array (it should be from buffer.add_transition)
        """
        self.network.eval()
        with torch.no_grad():
            input = torch.from_numpy(obs).float().unsqueeze(0)
            predicted_distance = self.network(input).squeeze(0).numpy()
            return predicted_distance.item()    # return scalar 
                
    def predict_distance_batch(self, obs_batch):
        # obs: (N,3) -> dist: (N,)
        self.network.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs_batch).float()
            y = self.network(x).squeeze(1)   # squeeze: (N,1) --> (N,)
            return y.numpy().astype(np.float32)    
        