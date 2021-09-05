import random
from collections import namedtuple
import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding
import math
import sys
sys.path.append("C:\\Users\\samatya.ASURITE\\PycharmProjects\\SocialGracefullnessTIV")
from constants import CONSTANTS as C
from autonomous_vehicle_nosim import AutonomousVehicle
import pickle


class Intent_Inference_Env(gym.Env):
    """
    Description:
    Two agents are in the intersection environment. The two vehicles start at the beginning of their respective tracks,
    and the goal is for both agents to cross the intersection while performing intent inference only when required.

    Source:
    The environment corresponds to the verion of the uncontrolled intersection problem described by Wang, Elliot, Ren,
     and Zhang in from TIV paper.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8911426

    Observations:
        Type: Box(4)??
        Num     Observation               Min                     Max
        0       Car Position              -2                       2
        1       Car Velocity               -0.05                   0.05
        2       Car Input                  -2                       3
        3       Belief theta Vehicles      0                       1      *4*2

    Actions:
        Type: Discrete(2)
        Num Action
        0   Do not carry out intent inference
        1   Carry out intent inference

    Reward:
        Reward is computed based on the reward of the task and the cost of carrying out intent inference at every step

    Starting state:
        All observations should be assigned a uniform random value in [-0.05.. 0.05]

    Episode termination:
        The episode terminates longer than 100 steps
        Solved requirements:
        Considered solved when both agents cross the intersection with out collision??


     """

    environment_name = "Intent Inference Environment"


    def __init__(self):

        #only using continious spaces here
        self.x_threshold = 3.0
        self.y_threshold = 3.0


        # is still within bounds. # car1 -2-1 # car2 2, -1
        low = np.array(
            [
                -2,
                -1,
                -0.05,
                -0.05,
                -2,
                -2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            dtype = np.float32,
        )

        high = np.array(
            [
                1,
                2,
                0.05,
                0.05,
                3,
                3,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2) # or 2, not sure
        self.observation_space = spaces.Box(low, high, dtype= np.float32)
        self.seed()
        self.reward_threshold = float("inf") #0.0
        self.trials = 100
        self._max_episode_steps = 100 #120
        self.id = "Intent Inf"
        self.action_translation = {0: "non inference", 1: "inference"}
        self.visited_final_state = False
        #set of parameters from autonomous vehicles
        self.P = C.PARAMETERSET_2
        #only car_1 will do empathetic intent inference, other car will carry on doing empathetic intent ??
        self.car_1 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_1,
                                       loss_style="reactive",
                                       who=1,
                                       inference_type="empathetic")  #M
        self.car_2 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_2,
                                       loss_style="reactive",
                                       who=0,
                                       inference_type="empathetic")  #H

        self.car_1.other_car = self.car_2
        self.car_2.other_car = self.car_1
        self.car_1.states_o = self.car_2.states
        self.car_2.states_o = self.car_1.states
        self.car_1.actions_set_o = self.car_2.actions_set
        self.car_2.actions_set_o = self.car_1.actions_set
        self.car_1.does_inference= True
        self.car_2.does_inference= True
        self.episode_steps = 0
        self.trial = 0


        # self.num_states = num_states
        #self.stochasticity_of_action_right = stochasticity_of_action_right
        # self.reward_if_visited_final_state = 1.0
        # self.reward_if_havent_visited_final_state = 0.01


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        One of the agents will continue intent inference while the other other one will do intermittent intent inferece
        :param action: [0, 1]
        :return: reward of both agents
        '''

        if type(action) is np.ndarray:
            action = action[0]
        assert action in [0, 1], "Action must be a 0 or a 1"

        skip_update_car2 = False


        #get actions here
        if action == 1: skip_update_car1 = False
        else: skip_update_car1 = True
        # if self.episode_steps == 0:
        #     skip_update_car1 = False



        self.car_1.update(self.episode_steps, skip_update_car1) # car 1 is machine ego vehicle
        self.car_2.update(self.episode_steps, skip_update_car2) # car 2 is human other vehicle
        self.episode_steps += 1

        #update for the other agent regardless
        #self.update_done_reward()
        intent_loss_car_1 = self.car_1.intent * np.exp(
            C.EXPTHETA * (- self.car_1.temp_action_set[C.ACTION_TIMESTEPS - 1][0] + 0.6))
        intent_loss_car_2 = self.car_2.intent * np.exp(
            C.EXPTHETA * (self.car_2.temp_action_set[C.ACTION_TIMESTEPS - 1][1] + 0.6))
        #D = np.sqrt(self.car_1.states[-1][0] * self.car_1.states[-1][0] + self.car_2.states[-1][1] * self.car_2.states[-1][1])
        predicted_distance = np.sum((self.car_1.temp_action_set - self.car_2.temp_action_set)**2, axis=1)
        D= np.sqrt(predicted_distance)

        #collision_loss = np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5))
        collision_loss = np.sum(np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5)))
        plannedloss_car1 = intent_loss_car_1 + collision_loss
        plannedloss_car2 = intent_loss_car_2 + collision_loss

        #reward = plannedloss_car1+ plannedloss_car2 - action*plannedloss_car1 #cumululative loss - effort
        alpha = 1
        reward = -(plannedloss_car1 + action * plannedloss_car1 * alpha)
        #reward = plannedloss_car1-action*plannedloss_car1 #car1 loss -effort
        #reward = plannedloss_car1 - action * (plannedloss_car1)/2

        self.state = (self.car_1.states[self.episode_steps][0], self.car_2.states[self.episode_steps][1],
                      self.car_1.actions_set[-1][0], self.car_2.actions_set[-1][1],
                      self.car_1.planned_trajectory[0], self.car_2.planned_trajectory[0],
                      self.car_1.joint_probability_matrix[0, 0], self.car_1.joint_probability_matrix[0, 1],
                      self.car_1.joint_probability_matrix[1, 0], self.car_1.joint_probability_matrix[1, 1],
                      self.car_2.joint_probability_matrix[0, 0], self.car_2.joint_probability_matrix[0, 1],
                      self.car_2.joint_probability_matrix[1, 0], self.car_2.joint_probability_matrix[1, 1])
        # self.state = (self.car_1.states[0][0], self.car_2.states[0][1], self.car_1.planned_trajectory[0],
        #               self.car_2.planned_trajectory[0], self.car_1.joint_probability_matrix[0,0], self.car_1.joint_probability_matrix[0,1],
        #               self.car_1.joint_probability_matrix[1, 0], self.car_1.joint_probability_matrix[1,1],
        #               self.car_2.joint_probability_matrix[0, 0], self.car_2.joint_probability_matrix[0, 1],
        #               self.car_2.joint_probability_matrix[1, 0], self.car_2.joint_probability_matrix[1, 1])


        self.s = self.state

        #threshold when task is done
        self.done = bool(
            self.car_1.states[self.episode_steps][0] < -2.0
            or self.car_1.states[self.episode_steps][0] > 1.0
            or self.car_2.states[self.episode_steps][1] < -1.0
            or self.car_2.states[self.episode_steps][1] > 2.0
            or self.episode_steps > self._max_episode_steps
            )



        # make an array of the states

        return self.s, reward, self.done, {} # states, reward, done or not

    def reset(self):
        #self.state = None # take environemnt from configuration
        self.next_state = None
        self.reward = None
        self.done = False
        self.visited_final_state = False
        self.episode_steps = 0

        import pickle

        data = pickle.load(open("uniform_data_dist.p", "rb"))
        xpos = data["si"][self.trial]
        ypos = data["sj"][self.trial]
        vi = data["vi"] [self.trial]
        vj = data["vj"][self.trial]
        # ui = data["ui"] [self.trial]
        # uj = data["uj"][self.trial]
        bji_1 = data["bji"][self.trial][0]
        bji_2 = data["bji"][self.trial][1]
        bji_3 = data["bji"][self.trial][2]
        bji_4 = data["bji"][self.trial][3]
        bij_1 = data["bij"][self.trial][0]
        bij_2 = data["bij"][self.trial][1]
        bij_3 = data["bij"][self.trial][2]
        bij_4 = data["bij"][self.trial][3]
        agg = data["agg"][self.trial]


        self.P.CAR_1.INITIAL_POSITION = np.array([xpos, 0]) #np.array([-2.0, 0])
        self.P.CAR_2.INITIAL_POSITION = np.array([0, ypos])
        C.PARAMETERSET_2.INITIAL_SPEED_1 = vi
        C.PARAMETERSET_2.INITIAL_SPEED_2 = vj
        #keeping initial velocity parameter constant for now
        # C.PARAMETERSET_2.INITIAL_SPEED_1 = vi
        # C.PARAMETERSET_2.INITIAL_SPEED_2 = vj

        if agg:
            self.P.CAR_2.INTENT = 1e6
        else:
            self.P.CAR_2.INTENT = 1



        # prev_ui = C.TRAJECTORY_SET #pick random
        # prev_uj = C.TRAJECTORY_SET #pick random

        #generate 4 random points for joint probability matrix



        self.car_1 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_1, # position from here
                                       loss_style="reactive",
                                       who=1,
                                       inference_type="empathetic")  #M
        self.car_2 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_2,
                                       loss_style="reactive",
                                       who=0,
                                       inference_type="non empathetic")  #H
        self.car_1.other_car = self.car_2
        self.car_2.other_car = self.car_1
        self.car_1.states_o = self.car_2.states
        self.car_2.states_o = self.car_1.states
        self.car_1.actions_set_o = self.car_2.actions_set
        self.car_2.actions_set_o = self.car_1.actions_set
        self.car_1.does_inference= True
        self.car_2.does_inference= True


        self.state = (self.car_1.states[0][0], self.car_2.states[0][1],
                      C.PARAMETERSET_2.INITIAL_SPEED_1, -C.PARAMETERSET_2.INITIAL_SPEED_2,
                      0 , 0,
                      bji_1, bji_2,
                      bji_3, bji_4,
                      bij_1, bij_2,
                      bij_3, bij_4)

        # self.state = (self.car_1.states[0][0], self.car_2.states[0][1],
        #               C.PARAMETERSET_2.INITIAL_SPEED_1, -C.PARAMETERSET_2.INITIAL_SPEED_2,
        #               0 , 0,
        #               self.car_1.joint_probability_matrix[0, 0], self.car_1.joint_probability_matrix[0, 1],
        #               self.car_1.joint_probability_matrix[1, 0], self.car_1.joint_probability_matrix[1, 1],
        #               self.car_2.joint_probability_matrix[0, 0], self.car_2.joint_probability_matrix[0, 1],
        #               self.car_2.joint_probability_matrix[1, 0], self.car_2.joint_probability_matrix[1, 1])


        # calculate self.theta based on numbers above for car 1 and car2
        if self.P.CAR_1.INTENT == 1:
            cap = [bji_1,bji_2]
            cap = np.array(cap)
            cap2 = cap/sum(cap)
            self.car_1.theta_probability = cap2
        else:
            cap = [bji_3, bji_4]
            cap = np.array(cap)
            cap2 = cap/sum(cap)
            self.car_1.theta_probability = cap2


        if self.P.CAR_2.INTENT == 1:
            cap = [bij_1,bij_2]
            cap = np.array(cap)
            cap2 = cap/sum(cap)
            self.car_2.theta_probability = cap2
        else:
            cap = [bij_3, bij_4]
            cap = np.array(cap)
            cap2 = cap/sum(cap)
            self.car_2.theta_probability = cap2


        self.s = np.array(self.state)
        self.trial = self.trial+1
        return self.s

    # def calc_next_states(self):


    # def update_done_reward(self):
    #     if self.next_state_x == 2 and self.next_state_y == 2:
    #         self.done = True
    #     if self.episode_steps >= self.max_episode_steps: self.done = True
    #
    #     self.reward = self.sum_of_reward()+ self.calc_effort()
    #
    #
    # def sum_of_reward(self):
    #     return 0
    #
    # def calc_effort(self):\S
    #     return 0

