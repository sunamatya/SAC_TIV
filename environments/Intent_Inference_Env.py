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

from sim_draw import Sim_Draw
from sim_data import Sim_Data
import pygame as pg
import datetime
import os


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
        ??

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

        self.sim_data = Sim_Data()

        if C.DRAW:
            self.sim_draw = Sim_Draw(self.P, C.ASSET_LOCATION)
            pg.display.flip()
            # self.capture = True if input("Capture video (y/n): ") else False
            self.capture = False
            self.output_data_pickle = False
            output_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            if self.output_data_pickle or self.capture:
                os.makedirs("./sim_outputs/%s" % output_name)
            if self.output_data_pickle:
                self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")

            if self.capture:
                self.output_dir = "./sim_outputs/%s/video/" % output_name
                os.makedirs(self.output_dir)


        #do not uncomment it
        # self.num_states = num_states
        # self.stochasticity_of_action_right = stochasticity_of_action_right
        # self.reward_if_visited_final_state = 1.0
        # self.reward_if_havent_visited_final_state = 0.01

        #figures printing and saving video
        self.show_prob_theta = True
        self.show_states = True
        self.show_action = True
        self.show_trajectory = True
        self.show_loss = True
        self.show_predicted_states_others = True
        self.show_does_inference = True


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
        if self.episode_steps == 0:
            skip_update_car1 = False



        self.car_1.update(self.episode_steps, skip_update_car1) # car 1 is machine ego vehicle
        self.car_2.update(self.episode_steps, skip_update_car2) # car 2 is human other vehicle
        self.episode_steps += 1

        #update for the other agent regardless
        #self.update_done_reward()
        intent_loss_car_1 = self.car_1.intent * np.exp(
            C.EXPTHETA * (- self.car_1.temp_action_set[C.ACTION_TIMESTEPS - 1][0] + 0.6))
        intent_loss_car_2 = self.car_2.intent * np.exp(
            C.EXPTHETA * (- self.car_2.temp_action_set[C.ACTION_TIMESTEPS - 1][0] + 0.6))
        D = np.sqrt(np.sum((np.array(self.car_1.states) - np.array(
            self.car_2.states)) ** 2)) + 1e-12  # np.sum((my_pos - other_pos)**2, axis=1)
        collision_loss = np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5))
        plannedloss_car1 = intent_loss_car_1 + collision_loss
        plannedloss_car2 = intent_loss_car_2 + collision_loss

        #reward = plannedloss_car1+ plannedloss_car2 - action*plannedloss_car1
        #reward = -(plannedloss_car1 + plannedloss_car2 + action * plannedloss_car1)
        reward = plannedloss_car1-action*plannedloss_car1
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
            self.car_1.states[0][0] < -self.x_threshold
            or self.car_1.states[0][0] > self.x_threshold
            or self.car_2.states[0][1] < -self.y_threshold
            or self.car_2.states[0][1] > self.y_threshold
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

        self.state = (self.car_1.states[0][0], self.car_2.states[0][1],
                      C.PARAMETERSET_2.INITIAL_SPEED, -C.PARAMETERSET_2.INITIAL_SPEED,
                      0 , 0,
                      self.car_1.joint_probability_matrix[0, 0], self.car_1.joint_probability_matrix[0, 1],
                      self.car_1.joint_probability_matrix[1, 0], self.car_1.joint_probability_matrix[1, 1],
                      self.car_2.joint_probability_matrix[0, 0], self.car_2.joint_probability_matrix[0, 1],
                      self.car_2.joint_probability_matrix[1, 0], self.car_2.joint_probability_matrix[1, 1])

        # self.state = (self.car_1.states[0][0], self.car_2.states[0][1], self.car_1.planned_trajectory[0][0],
        #               self.car_2.planned_trajectory[0][0], self.car_1.joint_probability_matrix[0,0], self.car_1.joint_probability_matrix[0,1],
        #               self.car_1.joint_probability_matrix[1, 0], self.car_1.joint_probability_matrix[1,1],
        #               self.car_2.joint_probability_matrix[0, 0], self.car_2.joint_probability_matrix[0, 1],
        #               self.car_2.joint_probability_matrix[1, 0], self.car_2.joint_probability_matrix[1, 1])


        self.s = np.array(self.state)
        return self.s

    def save_show_data(self):
        import numpy as np
        intent_loss_car_1 = self.car_1.intent * np.exp(
            C.EXPTHETA * (- self.car_1.temp_action_set[C.ACTION_TIMESTEPS - 1][0] + 0.6))
        intent_loss_car_2 = self.car_2.intent * np.exp(
            C.EXPTHETA * (- self.car_2.temp_action_set[C.ACTION_TIMESTEPS - 1][0] + 0.6))
        D = np.sqrt(np.sum((np.array(self.car_1.states) - np.array(
            self.car_2.states)) ** 2)) + 1e-12  # np.sum((my_pos - other_pos)**2, axis=1)
        collision_loss = np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5))
        plannedloss_car1 = intent_loss_car_1 + collision_loss
        plannedloss_car2 = intent_loss_car_2 + collision_loss

        self.sim_data.append_car1(states=self.car_1.states,
                                  actions=self.car_1.actions_set,
                                  action_sets=self.car_1.planned_actions_set,
                                  trajectory=self.car_1.planned_trajectory_set,
                                  predicted_theta_other=self.car_1.predicted_theta_other,
                                  predicted_theta_self=self.car_1.predicted_theta_self,
                                  predicted_actions_other=self.car_1.predicted_actions_other,
                                  predicted_others_prediction_of_my_actions=
                                  self.car_1.predicted_others_prediction_of_my_actions,
                                  wanted_trajectory_self=self.car_1.wanted_trajectory_self,
                                  wanted_trajectory_other=self.car_1.wanted_trajectory_other,
                                  wanted_states_other=self.car_1.wanted_states_other,
                                  inference_probability=self.car_1.inference_probability,
                                  inference_probability_proactive=self.car_1.inference_probability_proactive,
                                  theta_probability=self.car_1.theta_probability,
                                  social_gracefulness=self.car_1.social_gracefulness,
                                  planned_loss=plannedloss_car1,
                                  does_inf=self.car_1.skip_update,
                                  predicted_trajectory_other=self.car_1.predicted_trajectory_set_other)

        self.sim_data.append_car2(states=self.car_2.states,
                                  actions=self.car_2.actions_set,
                                  action_sets=self.car_2.planned_actions_set,
                                  trajectory=self.car_2.planned_trajectory_set,
                                  predicted_theta_other=self.car_2.predicted_theta_other,
                                  predicted_theta_self=self.car_2.predicted_theta_self,
                                  predicted_actions_other=self.car_2.predicted_actions_other,
                                  predicted_others_prediction_of_my_actions=
                                  self.car_2.predicted_others_prediction_of_my_actions,
                                  wanted_trajectory_self=self.car_2.wanted_trajectory_self,
                                  wanted_trajectory_other=self.car_2.wanted_trajectory_other,
                                  wanted_states_other=self.car_2.wanted_states_other,
                                  inference_probability=self.car_2.inference_probability,
                                  inference_probability_proactive=self.car_2.inference_probability_proactive,
                                  theta_probability=self.car_2.theta_probability,
                                  planned_loss=plannedloss_car2,
                                  does_inf=self.car_2.skip_update,
                                  predicted_trajectory_other=self.car_2.predicted_trajectory_set_other)

        self.sim_draw.draw_frame(self.sim_data, 0 , self.episode_steps-1)

    def show_plots(self):
        import matplotlib.pyplot as plt
        import numpy as np
        self.frame = self.episode_steps-1
        if self.show_prob_theta:
            car_1_theta = np.empty((0, 2))
            car_2_theta = np.empty((0, 2))
            for t in range(self.frame):
                car_1_theta = np.append(car_1_theta, np.expand_dims(self.sim_data.car2_theta_probability[t], axis=0), axis=0)
                car_2_theta = np.append(car_2_theta, np.expand_dims(self.sim_data.car1_theta_probability[t], axis=0), axis=0)

            plt.subplot(2, 1, 1)
            plt.title("Probability graph of the vehicle")
            plt.plot(range(1,self.frame+1), car_1_theta[:,0], label = "$\hat{\Theta}_M$= 1" )
            plt.plot(range(1,self.frame+1), car_1_theta[:,1], label = "$\hat{\Theta}_M$= 10^3")
            plt.ylabel("$p(\hat{\Theta}_M)$")
            plt.xlabel("frame")
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(range(1,self.frame+1), car_2_theta[:,0], label = "$\hat{\Theta}_H$= 1" )
            plt.plot(range(1,self.frame+1), car_2_theta[:,1], label = "$\hat{\Theta}_H$= 10^3" )
            plt.ylabel("$p(\hat{\Theta}_H)$")
            plt.xlabel("frame")
            plt.legend()

            plt.show()
            # for i in range(1,self.frame+1):
            #     if car_2_theta[i,0] == 1:
            #         print(i)
            #         break
            #plt.savefig('saved_figure.png')
        if self.show_states:
            car_1_state = np.empty((0, 2))
            car_2_state = np.empty((0, 2))
            for t in range(self.frame):
                car_1_state = np.append(car_1_state, np.expand_dims(self.sim_data.car1_states[t], axis=0), axis=0)
                car_2_state = np.append(car_2_state, np.expand_dims(self.sim_data.car2_states[t], axis=0), axis=0)
            dist = np.sqrt(car_1_state[:,0] *car_1_state[:,0] + car_2_state[:,1] * car_2_state[:,1])

            # plt.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
            # plt.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H', linestyle='--')
            # plt.legend()

            fig1, (ax1, ax2, ax3) = plt.subplots(3) #3 rows
            fig1.suptitle('Euclidean distance and Agent States')
            ax1.plot(dist, label='car dist')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='distance')

            ax2.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='states')

            ax3.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H')
            ax3.legend()
            ax3.set(xlabel='time', ylabel='states')
            plt.show()

        if self.show_action:
            car_1_action = np.empty((0, 2))
            car_2_action = np.empty((0, 2))
            car_1_action_predicted = np.empty((0, 2))
            for t in range(self.frame):
                car_1_action = np.append(car_1_action, (np.expand_dims(self.sim_data.car1_actions[t+1], axis=0) - np.expand_dims(self.sim_data.car1_actions[t], axis=0)), axis=0)
                car_2_action = np.append(car_2_action, (np.expand_dims(self.sim_data.car2_actions[t+1], axis=0) - np.expand_dims(self.sim_data.car2_actions[t], axis=0)), axis=0)
                # car_1_action = np.append(car_1_action, np.expand_dims(self.sim_data.car1_actions[t], axis=0), axis=0)
                # car_2_action = np.append(car_2_action, np.expand_dims(self.sim_data.car2_actions[t], axis=0), axis=0)
                # car_1_action_predicted = np.append(car_1_action_predicted, np.expand_dims(self.sim_data.car1_predicted_others_prediction_of_my_actions[t], axis=0), axis=0 )
                # car_2_action_predicted = np.append(car_2_action_predicted, np.expand_dims(self.sim_data.car1_predicted_actions_other[t], axis=0), axis=0 )
            #dist = np.sqrt(car_1_state[:,0] *car_1_state[:,0] + car_2_state[:,1] * car_2_state[:,1])

            # plt.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
            # plt.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H', linestyle='--')
            # plt.legend()

            fig1, (ax1, ax2) = plt.subplots(2) #3 rows
            # fig1.suptitle('Euclidean distance and Agent States')
            # ax1.plot(dist, label='car dist')
            # ax1.legend()
            # ax1.set(xlabel='time', ylabel='distance')

            ax1.plot(range(1,self.frame+1), car_1_action[:,0], label='car 1 actual action')
            #ax1.plot(range(1,self.frame+1), car_1_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='action')

            ax2.plot(range(1,self.frame+1), car_2_action[:,0], label='car 2 actual action')
            #ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='action')
            #plt.show()

        if self.show_loss:
            car_1_loss = np.empty((0, 1))
            car_2_loss = np.empty((0, 1))

            for t in range(self.frame):
                #def calculate_instanteous_reactive_loss(self, theta_self, trajectory, trajectory_other, s_self, s_other,s, probability):
                #car_1_instant = calculate_instanteous_reactive_loss(self.sim_data.car_1_theta, self.sim_data.car1_trajectory[t])
                # car_2_instant = calculate_instanteous_reactive_loss()
                car_1_loss = np.append(car_1_loss, self.sim_data.car1_planned_loss[t])
                car_2_loss = np.append(car_2_loss, self.sim_data.car2_planned_loss[t])
                # car_1_action_predicted = np.append(car_1_action_predicted, np.expand_dims(self.sim_data.car1_predicted_others_prediction_of_my_actions[t], axis=0), axis=0 )
                # car_2_action_predicted = np.append(car_2_action_predicted, np.expand_dims(self.sim_data.car1_predicted_actions_other[t], axis=0), axis=0 )
            #dist = np.sqrt(car_1_state[:,0] *car_1_state[:,0] + car_2_state[:,1] * car_2_state[:,1])

            # plt.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
            # plt.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H', linestyle='--')
            # plt.legend()

            fig1, (ax1, ax2) = plt.subplots(2) #3 rows
            # fig1.suptitle('Euclidean distance and Agent States')
            # ax1.plot(dist, label='car dist')
            # ax1.legend()
            # ax1.set(xlabel='time', ylabel='distance')

            ax1.plot(range(1,self.frame+1), car_1_loss, label='car 1 loss')
            #ax1.plot(range(1,self.frame+1), car_1_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='instant loss')

            ax2.plot(range(1,self.frame+1), car_2_loss, label='car 2 loss')
            #ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='instant loss')
            plt.show()

        if self.show_does_inference:
            car_1_does_inference = np.empty((0, 1))
            car_2_does_inference = np.empty((0, 1))

            for t in range(self.frame):
                car_1_does_inference = np.append(car_1_does_inference, self.sim_data.car1_does_inference[t])
                car_2_does_inference = np.append(car_2_does_inference, self.sim_data.car2_does_inference[t])


            fig1, (ax1, ax2) = plt.subplots(2) #3 rows
            ax1.plot(range(1,self.frame+1), car_1_does_inference, label='car 1 inference')
            #ax1.plot(range(1,self.frame+1), car_1_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='does inference')

            ax2.plot(range(1,self.frame+1), car_2_does_inference, label='car 2 inference')
            #ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='does inference')
            plt.show()

        if self.show_trajectory:
            car_1_predicted_trajectory_1 = np.empty((0))
            car_1_predicted_trajectory_2 = np.empty((0))
            car_2_predicted_trajectory_1 = np.empty((0))
            car_2_predicted_trajectory_2 = np.empty((0))
            car_1_planned_trajectory = np.empty((0))
            car_2_planned_trajectory = np.empty((0))
            car_1_timestep_2 = np.empty((0))
            car_2_timestep_2 = np.empty((0))
            # # car_1_performed_trajectory = np.empty((0, 1))
            # # car_2_performed_trajectory = np.empth((0, 1))
            #
            for t in range(self.frame):
                car_1_predicted_trajectory_1 = np.append(car_1_predicted_trajectory_1, self.sim_data.car2_predicted_trajectory_other[0][t][0][0])
                car_1_planned_trajectory = np.append(car_1_planned_trajectory, self.sim_data.car1_planned_trajectory_set[t][0])
                if len(self.sim_data.car2_predicted_trajectory_other[0][t]) == 2:
                    car_1_predicted_trajectory_2 = np.append(car_1_predicted_trajectory_2,
                                                             self.sim_data.car2_predicted_trajectory_other[0][t][1][0])
                    car_1_timestep_2 = np.append(car_1_timestep_2, t)

                car_2_predicted_trajectory_1 = np.append(car_2_predicted_trajectory_1, self.sim_data.car1_predicted_trajectory_other[0][t][0][0])
                car_2_planned_trajectory = np.append(car_2_planned_trajectory,
                                                     self.sim_data.car2_planned_trajectory_set[t][0])
                if len(self.sim_data.car1_predicted_trajectory_other[0][t]) == 2:
                    car_2_predicted_trajectory_2 = np.append(car_2_predicted_trajectory_2,
                                                             self.sim_data.car1_predicted_trajectory_other[0][t][1][0])
                    car_2_timestep_2 = np.append(car_2_timestep_2, t)
                #car_1_performed_trajectory = np.append(car_1_performed_trajectory, self.sim_data.car1_planned_)

            fig1, (ax1, ax2) = plt.subplots(2)
            # ax1.plot(range(1,self.frame+1), self.sim_data.car2_predicted_trajectory_other[0][1: self.frame+1], label='predicted trajectory of car 1(1)')
            # ax1.plot(range(1, self.frame + 1), self.sim_data.car2_predicted_trajectory_other[1] [1: self.frame+1],
            #          label='predicted trajectory of car 1(2)')
            # ax1.plot(range(1,self.frame+1), self.sim_data.car1_planned_trajectory_set, label='actual trajectory of car 1')
            # ax1.legend()
            # ax1.set(xlabel='time', ylabel='trajectory')

            ax1.plot(range(0,self.frame), car_1_predicted_trajectory_1, label='predicted trajectory of car 1(1)')
            ax1.plot(car_1_timestep_2 ,car_1_predicted_trajectory_2,
                     label='predicted trajectory of car 1(2)')
            ax1.plot(range(0,self.frame), car_1_planned_trajectory, label='actual trajectory of car 1')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='trajectory')

            ax2.plot(range(0,self.frame), car_2_predicted_trajectory_1, label='predicted trajectory of car 2 (1)')
            ax2.plot(car_2_timestep_2, car_2_predicted_trajectory_2, label='predicted trajectory of car 2 (2)')
            ax2.plot(range(0,self.frame), car_2_planned_trajectory, label='actual trajectory of car 2')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='trajectory')
            plt.show()






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

