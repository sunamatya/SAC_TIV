import random
from collections import namedtuple
import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding
import math
import sys
sys.path.append("C:\\Users\\samatya.ASURITE\\PycharmProjects\\Game-Theoretic-Belief-Distribution")
from environment import Environment
from savi_simulation_no_draw import Simulation
import argparse
import os
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
        0       Car Position              15                       36
        1       Car Velocity              0                       30
        2       Car Input                 -5                      10
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

        # is still within bounds. # car1 -2-1 # car2 2, -1
        low = np.array(
            [
                15,
                15,
                0,
                0,
                -5,
                -5,
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
                37,
                37,
                30,
                30,
                10,
                10,
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
        self._max_episode_steps = 35 #120
        self.id = "Intent Inf"
        self.action_translation = {0: "non inference", 1: "inference"}
        self.visited_final_state = False
        #set of parameters from autonomous vehicles



        parser = argparse.ArgumentParser()
        """
        simulation parameters
        """
        parser.add_argument('--sim_duration', type=int, default=100)  # time span for simulation
        parser.add_argument('--sim_dt', type=int, default=0.05)  # time step in simulation: choices: [0.01, 0.25, 1]
        parser.add_argument('--sim_lr', type=float, default=0.1)  # learning rate
        parser.add_argument('--sim_nepochs', type=int, default=100)  # number of training epochs
        parser.add_argument('--save', type=str, default='./experiment')  # save dir
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--gpu', type=int, default=0)

        """
        environment parameters
        """
        parser.add_argument('--env_name', type=str, choices=['trained_intersection', 'bvp_intersection'],
                            default='bvp_intersection')
        # Starting position and velocity is set within the environment

        """
        agent model parameters
        """
        # TODO: bvp_continuous model is under development
        # choose inference model: use bvp for our experiment, and only for 1st player (i.e. ['bvp', 'none'])
        parser.add_argument('--agent_inference', type=str, choices=['none', 'bvp', 'bvp_continuous'],
                            default=['bvp', 'bvp'])  # ['bvp', 'none']

        # choose decision model: use the same model for the two agent, bvp_non_empathetic or bvp_empathetic
        parser.add_argument('--agent_decision', type=str,
                            choices=['constant_speed', 'bvp_baseline', 'bvp_optimize', 'bvp_optimize_costate',
                                     'bvp_non_empathetic', 'bvp_empathetic',
                                     'bvp_e_optimize', 'bvp_ne_optimize'],
                            default=['bvp_empathetic', 'bvp_empathetic'])

        """
        agent parameters (for the proposed s = <x0,p0(β),β†,∆t,l>), for 2 agent case
        """

        parser.add_argument('--agent_dt', type=int, default=1)  # time step in planning (NOT IN USE)
        parser.add_argument('--agent_intent', type=str, choices=['NA', 'A'],
                            default=['NA', 'A'])  # AGENT TRUE PARAM [P1, P2]
        parser.add_argument('--agent_noise', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
        parser.add_argument('--agent_intent_belief', type=str, choices=['NA', 'A'], default=['A', 'A'])  # AGENT BELIEF
        parser.add_argument('--agent_noise_belief', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
        parser.add_argument('--belief_weight', type=float, default=0.8)

        # parser.add_argument('', type=str, choices=[], default=[])
        args = parser.parse_args()

        initial_states = [[20, 18], [20, 18]]  # x1, v1, x2, v2


        if args.env_name == 'bvp_intersection':
            sim_par = {"theta": [5, 1],  # NA, A
                       "lambda": [0.1, 0.5],  # N, NN # [0.1, 0.5],
                       # "action_set": [-5, -3, -1, 0, 2, 4, 6, 8, 10],
                       "action_set": [-5, 0, 3, 7, 10],
                       # "action_set": close_action_set,
                       }
        elif args.env_name == 'trained_intersection':
            sim_par = {"theta": [1, 1000],  # NA, A
                       "lambda": [0.001, 0.005],  # N, NN
                       "action_set": [-8, -4, 0, 4, 8],
                       }
        else:
            sim_par = {"theta": [1, 1000],
                       "lambda": [0.001, 0.005],
                       "action_set": [-8, -4, 0, 4, 8],
                       }

        e = Environment(args.env_name, args.agent_inference, sim_par, initial_states, args.sim_dt, args.agent_intent,
                        args.agent_noise,
                        args.agent_intent_belief,
                        args.agent_noise_belief)
        assert len(args.agent_inference) == e.N_AGENTS and len(args.agent_decision) == e.N_AGENTS

        kwargs = {"env": e,
                  "duration": args.sim_duration,
                  "n_agents": e.N_AGENTS,
                  "inference_type": args.agent_inference,
                  "decision_type": args.agent_decision,
                  "sim_dt": args.sim_dt,
                  "sim_lr": args.sim_lr,
                  "sim_par": sim_par,
                  "sim_nepochs": args.sim_nepochs,
                  "belief_weight": args.belief_weight}
        self.s = Simulation(**kwargs) #this makes the two agent

        self.episode_steps = 0
        self.does_inference = []
        self.joint_probability_matrix1 = []
        self.joint_probability_matrix2 = []

        self.trial = 148



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

        # get actions here
        if action == 1:
            self.s.agents[0].env.agent_inference = "bvp"
        else:
            self.s.agents[0].env.agent_inference = "none"

        self.s.agents[0].update(self.s)

        self.s.agents[1].update(self.s)
        L1, L2 = self.s.calc_loss()
        reward = -(L1 + L2 + action * 400)

        # all the states
        curr_state_h = self.s.agents[0].state[self.s.frame]
        curr_state_m = self.s.agents[1].state[self.s.frame]
        action_set_0 = self.s.agents[0].action_bvp[-1]
        action_set_1 = self.s.agents[1].action_bvp[-1]

        self.episode_steps += 1
        self.s.frame = self.episode_steps
        self.does_inference.append(action)
        self.joint_probability_matrix1.append(self.s.agents[0].joint_probability_matrix)
        self.joint_probability_matrix2.append(self.s.agents[1].joint_probability_matrix)


        self.state = (curr_state_h[1], curr_state_m[0], curr_state_h[3], curr_state_m[2],
                      action_set_0[0], action_set_1[1],
                      self.s.agents[0].joint_probability_matrix[0, 0], self.s.agents[0].joint_probability_matrix[0, 1],
                      self.s.agents[0].joint_probability_matrix[1, 0], self.s.agents[0].joint_probability_matrix[1, 1],
                      self.s.agents[1].joint_probability_matrix[0, 0], self.s.agents[1].joint_probability_matrix[0, 1],
                      self.s.agents[1].joint_probability_matrix[1, 0], self.s.agents[1].joint_probability_matrix[1, 1])

        # threshold when task is done
        self.done = bool(
            curr_state_h[1] > 36
            or curr_state_m[0] > 36
            or self.episode_steps > self._max_episode_steps
        )

        # make an array of the states

        return self.state, reward, self.done, {}  # states, reward, done or not

    def reset(self):
        #self.state = None # take environemnt from configuration
        self.next_state = None
        self.reward = None
        self.done = False
        self.visited_final_state = False
        self.episode_steps = 0

        import pickle

        #data = pickle.load(open("uniform_data_dist.p", "rb"))
        #data = pickle.load(open("uniform_data_dist_3_3_200.p", "rb"))
        data = pickle.load(open("uniform_data_dist_bvp_200.p", "rb"))
        xpos = data["si"][self.trial]
        ypos = data["sj"][self.trial]
        belief_weight_no = data["wo"][self.trial]

        # xpos = 18
        # ypos = 18
        # belief_weight_no = 0.8




        parser = argparse.ArgumentParser()
        """
        simulation parameters
        """
        parser.add_argument('--sim_duration', type=int, default=100)  # time span for simulation
        parser.add_argument('--sim_dt', type=int, default=0.05)  # time step in simulation: choices: [0.01, 0.25, 1]
        parser.add_argument('--sim_lr', type=float, default=0.1)  # learning rate
        parser.add_argument('--sim_nepochs', type=int, default=100)  # number of training epochs
        parser.add_argument('--save', type=str, default='./experiment')  # save dir
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--gpu', type=int, default=0)

        """
        environment parameters
        """
        parser.add_argument('--env_name', type=str, choices=['trained_intersection', 'bvp_intersection'],
                            default='bvp_intersection')
        # Starting position and velocity is set within the environment

        """
        agent model parameters
        """
        # TODO: bvp_continuous model is under development
        # choose inference model: use bvp for our experiment, and only for 1st player (i.e. ['bvp', 'none'])
        parser.add_argument('--agent_inference', type=str, choices=['none', 'bvp', 'bvp_continuous'],
                            default=['bvp', 'bvp'])  # ['bvp', 'none']

        # choose decision model: use the same model for the two agent, bvp_non_empathetic or bvp_empathetic
        parser.add_argument('--agent_decision', type=str,
                            choices=['constant_speed', 'bvp_baseline', 'bvp_optimize', 'bvp_optimize_costate',
                                     'bvp_non_empathetic', 'bvp_empathetic',
                                     'bvp_e_optimize', 'bvp_ne_optimize'],
                            default=['bvp_empathetic', 'bvp_empathetic'])

        """
        agent parameters (for the proposed s = <x0,p0(β),β†,∆t,l>), for 2 agent case
        """

        parser.add_argument('--agent_dt', type=int, default=1)  # time step in planning (NOT IN USE)
        parser.add_argument('--agent_intent', type=str, choices=['NA', 'A'],
                            default=['NA', 'A'])  # AGENT TRUE PARAM [P1, P2]
        parser.add_argument('--agent_noise', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
        parser.add_argument('--agent_intent_belief', type=str, choices=['NA', 'A'], default=['A', 'A'])  # AGENT BELIEF
        parser.add_argument('--agent_noise_belief', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
        parser.add_argument('--belief_weight', type=float, default=0.8)

        # parser.add_argument('', type=str, choices=[], default=[])
        args = parser.parse_args()

        initial_states = [[xpos, 18], [ypos, 18]]  # x1, v1, x2, v2


        if args.env_name == 'bvp_intersection':
            sim_par = {"theta": [5, 1],  # NA, A
                       "lambda": [0.1, 0.5],  # N, NN # [0.1, 0.5],
                       # "action_set": [-5, -3, -1, 0, 2, 4, 6, 8, 10],
                       "action_set": [-5, 0, 3, 7, 10],
                       # "action_set": close_action_set,
                       }
        elif args.env_name == 'trained_intersection':
            sim_par = {"theta": [1, 1000],  # NA, A
                       "lambda": [0.001, 0.005],  # N, NN
                       "action_set": [-8, -4, 0, 4, 8],
                       }
        else:
            sim_par = {"theta": [1, 1000],
                       "lambda": [0.001, 0.005],
                       "action_set": [-8, -4, 0, 4, 8],
                       }

        e = Environment(args.env_name, args.agent_inference, sim_par, initial_states, args.sim_dt, args.agent_intent,
                        args.agent_noise,
                        args.agent_intent_belief,
                        args.agent_noise_belief)
        assert len(args.agent_inference) == e.N_AGENTS and len(args.agent_decision) == e.N_AGENTS

        kwargs = {"env": e,
                  "duration": args.sim_duration,
                  "n_agents": e.N_AGENTS,
                  "inference_type": args.agent_inference,
                  "decision_type": args.agent_decision,
                  "sim_dt": args.sim_dt,
                  "sim_lr": args.sim_lr,
                  "sim_par": sim_par,
                  "sim_nepochs": args.sim_nepochs,
                  "belief_weight": belief_weight_no}
        self.s = Simulation(**kwargs) #this makes the two agent

        joint_probability_matrix = np.zeros((2,2))
        joint_probability_matrix2 = np.zeros((2, 2))
        p_beta_d = self.s.initial_belief
        joint_probability_matrix[0, 0] = np.sum(p_beta_d[0:2, 0:2])
        joint_probability_matrix[0, 1] = np.sum(p_beta_d[0:2, 2:4])
        joint_probability_matrix[1, 0] = np.sum(p_beta_d[2:4, 0:2])
        joint_probability_matrix[1, 1] = np.sum(p_beta_d[2:4, 2:4])

        p_beta_d = self.s.initial_belief
        joint_probability_matrix2[0, 0] = np.sum(p_beta_d[0:2, 0:2])
        joint_probability_matrix2[0, 1] = np.sum(p_beta_d[0:2, 2:4])
        joint_probability_matrix2[1, 0] = np.sum(p_beta_d[2:4, 0:2])
        joint_probability_matrix2[1, 1] = np.sum(p_beta_d[2:4, 2:4])

        self.state = (xpos, ypos, 18, 18,
                      0, 0,
                      joint_probability_matrix[0, 0], joint_probability_matrix[0, 1],
                      joint_probability_matrix[1, 0], joint_probability_matrix[1, 1],
                      joint_probability_matrix2[0, 0], joint_probability_matrix2[0, 1],
                      joint_probability_matrix2[1, 0], joint_probability_matrix2[1, 1])

        output_name = "data_trial_" + str(self.trial % 200)

        if not os.path.exists("./sim_outputs/%s" % output_name):
            os.makedirs("./sim_outputs/%s" % output_name)

        self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")
        self.trial = self.trial+1
        self.does_inference = []
        return np.array(self.state)



    def dump_data(self):

        data= {}
        agent1 = {}
        agent2 = {}
        key = ['state',  # actual states
               'action',  # actual actions (converted from trajectory)
               'last_actions',  # last actions for decision/inference purposes
               'planned_trajectory_set',  # future state trajectories based on the planned actions
               'predicted_actions_other',  # other's future actions, estimated
               'predicted_actions_self',  # other's belief of self future actions, estimated
               'predicted_states_other',  # prediction of other's future states
               'predicted_states_self',  # prediction of other's future states
               'predicted_intent_all',  # calculated joint probabilities of intent and rationality for all agents
               'predicted_intent_other',  # calculated joint probabilities of intent and rationality
               'predicted_intent_self',  # calculated joint probabilities of intent and rationality
               'predicted_policy_other',
               'belief_count',  # count number of times param has the highest probability mass
               'policy_choice',  # records the choice of policy made at each time step
               ]

        for k in key:
            agent1[k] = getattr(self.s.agents[0], k)
            agent2[k] = getattr(self.s.agents[1], k)

        data["agent1"] = agent1
        data["agent2"] = agent2
        data["does_inference"] = self.does_inference



        pickle.dump(data, self.sim_out, pickle.HIGHEST_PROTOCOL)
        print('Output pickled and dumped.')

    # def show_plots(self):
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     self.frame = self.episode_steps-1
    #     if self.show_prob_theta:
    #         # car_1_theta = np.empty((0, 2))
    #         # car_2_theta = np.empty((0, 2))
    #         car_1_theta = np.ones((1,2))*0.5
    #         car_2_theta = np.ones((1,2))*0.5
    #
    #         for t in range(self.frame):
    #             car_1_theta = np.append(car_1_theta, np.expand_dims(self.sim_data.car2_theta_probability[t], axis=0), axis=0)
    #             car_2_theta = np.append(car_2_theta, np.expand_dims(self.sim_data.car1_theta_probability[t], axis=0), axis=0)
    #
    #         plt.subplot(2, 1, 1)
    #         plt.title("Probability graph of the vehicle")
    #         plt.plot(range(0,self.frame+1), car_1_theta[:,0], label = "$\hat{\Theta}_M$= 1" )
    #         plt.plot(range(0,self.frame+1), car_1_theta[:,1], label = "$\hat{\Theta}_M$= 10^3")
    #         plt.ylabel("$p(\hat{\Theta}_M)$")
    #         plt.xlabel("frame")
    #         plt.legend()
    #         plt.subplot(2, 1, 2)
    #         plt.plot(range(0,self.frame+1), car_2_theta[:,0], label = "$\hat{\Theta}_H$= 1" )
    #         plt.plot(range(0,self.frame+1), car_2_theta[:,1], label = "$\hat{\Theta}_H$= 10^3" )
    #         plt.ylabel("$p(\hat{\Theta}_H)$")
    #         plt.xlabel("frame")
    #         plt.legend()
    #
    #         plt.show()
    #         # for i in range(1,self.frame+1):
    #         #     if car_2_theta[i,0] == 1:
    #         #         print(i)
    #         #         break
    #         #plt.savefig('saved_figure.png')
    #     if self.show_states:
    #         car_1_state = np.empty((0, 2))
    #         car_2_state = np.empty((0, 2))
    #         for t in range(self.frame):
    #             car_1_state = np.append(car_1_state, np.expand_dims(self.sim_data.car1_states[t], axis=0), axis=0)
    #             car_2_state = np.append(car_2_state, np.expand_dims(self.sim_data.car2_states[t], axis=0), axis=0)
    #         dist = np.sqrt(car_1_state[:,0] *car_1_state[:,0] + car_2_state[:,1] * car_2_state[:,1])
    #
    #         # plt.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
    #         # plt.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H', linestyle='--')
    #         # plt.legend()
    #
    #         fig1, (ax1, ax2, ax3) = plt.subplots(3) #3 rows
    #         fig1.suptitle('Euclidean distance and Agent States')
    #         ax1.plot(dist, label='car dist')
    #         ax1.legend()
    #         ax1.set(xlabel='time', ylabel='distance')
    #
    #         ax2.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
    #         ax2.legend()
    #         ax2.set(xlabel='time', ylabel='states')
    #
    #         ax3.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H')
    #         ax3.legend()
    #         ax3.set(xlabel='time', ylabel='states')
    #         plt.show()
    #
    #     if self.show_action:
    #         car_1_action = np.empty((0, 2))
    #         car_2_action = np.empty((0, 2))
    #         car_1_action_predicted = np.empty((0, 2))
    #         for t in range(self.frame):
    #             car_1_action = np.append(car_1_action, (np.expand_dims(self.sim_data.car1_actions[t+1], axis=0) - np.expand_dims(self.sim_data.car1_actions[t], axis=0)), axis=0)
    #             car_2_action = np.append(car_2_action, (np.expand_dims(self.sim_data.car2_actions[t+1], axis=0) - np.expand_dims(self.sim_data.car2_actions[t], axis=0)), axis=0)
    #             # car_1_action = np.append(car_1_action, np.expand_dims(self.sim_data.car1_actions[t], axis=0), axis=0)
    #             # car_2_action = np.append(car_2_action, np.expand_dims(self.sim_data.car2_actions[t], axis=0), axis=0)
    #             # car_1_action_predicted = np.append(car_1_action_predicted, np.expand_dims(self.sim_data.car1_predicted_others_prediction_of_my_actions[t], axis=0), axis=0 )
    #             # car_2_action_predicted = np.append(car_2_action_predicted, np.expand_dims(self.sim_data.car1_predicted_actions_other[t], axis=0), axis=0 )
    #         #dist = np.sqrt(car_1_state[:,0] *car_1_state[:,0] + car_2_state[:,1] * car_2_state[:,1])
    #
    #         # plt.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
    #         # plt.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H', linestyle='--')
    #         # plt.legend()
    #
    #         fig1, (ax1, ax2) = plt.subplots(2) #3 rows
    #         # fig1.suptitle('Euclidean distance and Agent States')
    #         # ax1.plot(dist, label='car dist')
    #         # ax1.legend()
    #         # ax1.set(xlabel='time', ylabel='distance')
    #
    #         ax1.plot(range(1,self.frame+1), car_1_action[:,0], label='car 1 actual action')
    #         #ax1.plot(range(1,self.frame+1), car_1_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
    #         ax1.legend()
    #         ax1.set(xlabel='time', ylabel='action')
    #
    #         ax2.plot(range(1,self.frame+1), car_2_action[:,0], label='car 2 actual action')
    #         #ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
    #         ax2.legend()
    #         ax2.set(xlabel='time', ylabel='action')
    #         #plt.show()
    #
    #     if self.show_loss:
    #         car_1_loss = np.empty((0, 1))
    #         car_2_loss = np.empty((0, 1))
    #
    #         for t in range(self.frame):
    #             #def calculate_instanteous_reactive_loss(self, theta_self, trajectory, trajectory_other, s_self, s_other,s, probability):
    #             #car_1_instant = calculate_instanteous_reactive_loss(self.sim_data.car_1_theta, self.sim_data.car1_trajectory[t])
    #             # car_2_instant = calculate_instanteous_reactive_loss()
    #             car_1_loss = np.append(car_1_loss, self.sim_data.car1_planned_loss[t])
    #             car_2_loss = np.append(car_2_loss, self.sim_data.car2_planned_loss[t])
    #             # car_1_action_predicted = np.append(car_1_action_predicted, np.expand_dims(self.sim_data.car1_predicted_others_prediction_of_my_actions[t], axis=0), axis=0 )
    #             # car_2_action_predicted = np.append(car_2_action_predicted, np.expand_dims(self.sim_data.car1_predicted_actions_other[t], axis=0), axis=0 )
    #         #dist = np.sqrt(car_1_state[:,0] *car_1_state[:,0] + car_2_state[:,1] * car_2_state[:,1])
    #
    #         # plt.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
    #         # plt.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H', linestyle='--')
    #         # plt.legend()
    #
    #         fig1, (ax1, ax2) = plt.subplots(2) #3 rows
    #         # fig1.suptitle('Euclidean distance and Agent States')
    #         # ax1.plot(dist, label='car dist')
    #         # ax1.legend()
    #         # ax1.set(xlabel='time', ylabel='distance')
    #
    #         ax1.plot(range(1,self.frame+1), car_1_loss, label='car 1 loss')
    #         #ax1.plot(range(1,self.frame+1), car_1_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
    #         ax1.legend()
    #         ax1.set(xlabel='time', ylabel='instant loss')
    #
    #         ax2.plot(range(1,self.frame+1), car_2_loss, label='car 2 loss')
    #         #ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
    #         ax2.legend()
    #         ax2.set(xlabel='time', ylabel='instant loss')
    #         plt.show()
    #
    #     if self.show_does_inference:
    #         car_1_does_inference = np.empty((0, 1))
    #         car_2_does_inference = np.empty((0, 1))
    #
    #         for t in range(self.frame):
    #             car_1_does_inference = np.append(car_1_does_inference, self.sim_data.car1_does_inference[t])
    #             car_2_does_inference = np.append(car_2_does_inference, self.sim_data.car2_does_inference[t])
    #
    #
    #         fig1, (ax1, ax2) = plt.subplots(2) #3 rows
    #         ax1.plot(range(1,self.frame+1), car_1_does_inference, label='car 1 inference')
    #         #ax1.plot(range(1,self.frame+1), car_1_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
    #         ax1.legend()
    #         ax1.set(xlabel='time', ylabel='does inference')
    #
    #         ax2.plot(range(1,self.frame+1), car_2_does_inference, label='car 2 inference')
    #         #ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
    #         ax2.legend()
    #         ax2.set(xlabel='time', ylabel='does inference')
    #         plt.show()
    #
    #     if self.show_trajectory:
    #         car_1_predicted_trajectory_1 = np.empty((0))
    #         car_1_predicted_trajectory_2 = np.empty((0))
    #         car_2_predicted_trajectory_1 = np.empty((0))
    #         car_2_predicted_trajectory_2 = np.empty((0))
    #         car_1_planned_trajectory = np.empty((0))
    #         car_2_planned_trajectory = np.empty((0))
    #         car_1_timestep_2 = np.empty((0))
    #         car_2_timestep_2 = np.empty((0))
    #         # # car_1_performed_trajectory = np.empty((0, 1))
    #         # # car_2_performed_trajectory = np.empth((0, 1))
    #         #
    #         for t in range(self.frame):
    #             car_1_predicted_trajectory_1 = np.append(car_1_predicted_trajectory_1, self.sim_data.car2_predicted_trajectory_other[0][t][0][0])
    #             car_1_planned_trajectory = np.append(car_1_planned_trajectory, self.sim_data.car1_planned_trajectory_set[t][0])
    #             if len(self.sim_data.car2_predicted_trajectory_other[0][t]) == 2:
    #                 car_1_predicted_trajectory_2 = np.append(car_1_predicted_trajectory_2,
    #                                                          self.sim_data.car2_predicted_trajectory_other[0][t][1][0])
    #                 car_1_timestep_2 = np.append(car_1_timestep_2, t)
    #
    #             car_2_predicted_trajectory_1 = np.append(car_2_predicted_trajectory_1, self.sim_data.car1_predicted_trajectory_other[0][t][0][0])
    #             car_2_planned_trajectory = np.append(car_2_planned_trajectory,
    #                                                  self.sim_data.car2_planned_trajectory_set[t][0])
    #             if len(self.sim_data.car1_predicted_trajectory_other[0][t]) == 2:
    #                 car_2_predicted_trajectory_2 = np.append(car_2_predicted_trajectory_2,
    #                                                          self.sim_data.car1_predicted_trajectory_other[0][t][1][0])
    #                 car_2_timestep_2 = np.append(car_2_timestep_2, t)
    #             #car_1_performed_trajectory = np.append(car_1_performed_trajectory, self.sim_data.car1_planned_)
    #
    #         fig1, (ax1, ax2) = plt.subplots(2)
    #         # ax1.plot(range(1,self.frame+1), self.sim_data.car2_predicted_trajectory_other[0][1: self.frame+1], label='predicted trajectory of car 1(1)')
    #         # ax1.plot(range(1, self.frame + 1), self.sim_data.car2_predicted_trajectory_other[1] [1: self.frame+1],
    #         #          label='predicted trajectory of car 1(2)')
    #         # ax1.plot(range(1,self.frame+1), self.sim_data.car1_planned_trajectory_set, label='actual trajectory of car 1')
    #         # ax1.legend()
    #         # ax1.set(xlabel='time', ylabel='trajectory')
    #
    #         ax1.plot(range(0,self.frame), car_1_predicted_trajectory_1, label='predicted trajectory of car 1(1)')
    #         ax1.plot(car_1_timestep_2 ,car_1_predicted_trajectory_2,
    #                  label='predicted trajectory of car 1(2)')
    #         ax1.plot(range(0,self.frame), car_1_planned_trajectory, label='actual trajectory of car 1')
    #         ax1.legend()
    #         ax1.set(xlabel='time', ylabel='trajectory')
    #
    #         ax2.plot(range(0,self.frame), car_2_predicted_trajectory_1, label='predicted trajectory of car 2 (1)')
    #         ax2.plot(car_2_timestep_2, car_2_predicted_trajectory_2, label='predicted trajectory of car 2 (2)')
    #         ax2.plot(range(0,self.frame), car_2_planned_trajectory, label='actual trajectory of car 2')
    #         ax2.legend()
    #         ax2.set(xlabel='time', ylabel='trajectory')
    #         plt.show()
    #
    #     if self.show_reward:
    #         plt.plot(self.reward_container)
    #         plt.show()
    #
    #     if self.capture:
    #         # Compile to video
    #         # os.system("ffmpeg -f image2 -framerate 1 -i %simg%%03d.jpeg %s/output_video.mp4 " % (self.output_dir, self.output_dir))
    #         img_list = [self.output_dir+"img"+str(i).zfill(3)+".jpeg" for i in range( self.episode_steps-1)]
    #         import imageio
    #         images = []
    #         for filename in img_list:
    #             images.append(imageio.imread(filename))
    #         imageio.mimsave(self.output_dir+'movie.gif', images)
    #         #
    #         # # Delete images
    #         # [os.remove(self.output_dir + file) for file in os.listdir(self.output_dir) if ".jpeg" in file]
    #         # print("Simulation video output saved to %s." % self.output_dir)
    #     print("Simulation ended.")





