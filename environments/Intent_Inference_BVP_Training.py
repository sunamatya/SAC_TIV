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
        parser.add_argument('--agent_intent', type=str, choices=['A', 'A'],
                            default=['A', 'A'])  # AGENT TRUE PARAM [P1, P2]
        parser.add_argument('--agent_noise', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
        parser.add_argument('--agent_intent_belief', type=str, choices=['NA', 'A'], default=['NA', 'NA'])  # AGENT BELIEF
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
        self.trial = 0

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

        #get actions here
        if action == 1:
            self.s.agents[0].env.agent_inference = "bvp"
        else:
            self.s.agents[0].env.agent_inference = "none"

        self.s.agents[0].update(self.s)
        self.s.agents[1].env.agent_inference = "bvp"
        self.s.agents[1].update(self.s)
        L1, L2 = self.s.calc_loss()
        reward = -(L1 + L2 + action * 400)



        #all the states
        curr_state_h = self.s.agents[0].state[self.s.frame]
        curr_state_m = self.s.agents[1].state[self.s.frame]
        action_set_0 = self.s.agents[0].action_bvp[-1]
        action_set_1 = self.s.agents[1].action_bvp[-1]


        self.episode_steps += 1
        self.s.frame = self.episode_steps


        self.state = (curr_state_h[1], curr_state_m[0], curr_state_h[3], curr_state_m[2],
                      action_set_0[0], action_set_1[1],
                      self.s.agents[0].joint_probability_matrix[0, 0], self.s.agents[0].joint_probability_matrix[0, 1],
                      self.s.agents[0].joint_probability_matrix[1, 0], self.s.agents[0].joint_probability_matrix[1, 1],
                      self.s.agents[1].joint_probability_matrix[0, 0], self.s.agents[1].joint_probability_matrix[0, 1],
                      self.s.agents[1].joint_probability_matrix[1, 0], self.s.agents[1].joint_probability_matrix[1, 1])

        #threshold when task is done
        self.done = bool(
            curr_state_h[1] > 36
            or curr_state_m[0] > 36
            or self.episode_steps > self._max_episode_steps
            )



        # make an array of the states

        return self.state, reward, self.done, {} # states, reward, done or not

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
        parser.add_argument('--agent_intent', type=str, choices=['A', 'A'],
                            default=['A', 'A'])  # AGENT TRUE PARAM [P1, P2]
        parser.add_argument('--agent_noise', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
        parser.add_argument('--agent_intent_belief', type=str, choices=['NA', 'A'], default=['NA', 'NA'])  # AGENT BELIEF
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

        # self.car_1 = AutonomousVehicle(scenario_parameters=self.P,
        #                                car_parameters_self=self.P.CAR_1, # position from here
        #                                loss_style="reactive",
        #                                who=1,
        #                                inference_type="empathetic")  #M
        # self.car_2 = AutonomousVehicle(scenario_parameters=self.P,
        #                                car_parameters_self=self.P.CAR_2,
        #                                loss_style="reactive",
        #                                who=0,
        #                                inference_type="empathetic")  #H
        # self.car_1.other_car = self.car_2
        # self.car_2.other_car = self.car_1
        # self.car_1.states_o = self.car_2.states
        # self.car_2.states_o = self.car_1.states
        # self.car_1.actions_set_o = self.car_2.actions_set
        # self.car_2.actions_set_o = self.car_1.actions_set
        # self.car_1.does_inference= True
        # self.car_2.does_inference= True


        # self.state = (self.car_1.states[0][0], self.car_2.states[0][1],
        #               C.PARAMETERSET_2.INITIAL_SPEED_1, -C.PARAMETERSET_2.INITIAL_SPEED_2,
        #               0 , 0,
        #               bji_1, bji_2,
        #               bji_3, bji_4,
        #               bij_1, bij_2,
        #               bij_3, bij_4)

        # self.state = (self.car_1.states[0][0], self.car_2.states[0][1],
        #               C.PARAMETERSET_2.INITIAL_SPEED_1, -C.PARAMETERSET_2.INITIAL_SPEED_2,
        #               0 , 0,
        #               self.car_1.joint_probability_matrix[0, 0], self.car_1.joint_probability_matrix[0, 1],
        #               self.car_1.joint_probability_matrix[1, 0], self.car_1.joint_probability_matrix[1, 1],
        #               self.car_2.joint_probability_matrix[0, 0], self.car_2.joint_probability_matrix[0, 1],
        #               self.car_2.joint_probability_matrix[1, 0], self.car_2.joint_probability_matrix[1, 1])


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


        # # calculate self.theta based on numbers above for car 1 and car2
        # if self.P.CAR_1.INTENT == 1:
        #     cap = [bji_1,bji_2]
        #     cap = np.array(cap)
        #     cap2 = cap/sum(cap)
        #     self.car_1.theta_probability = cap2
        # else:
        #     cap = [bji_3, bji_4]
        #     cap = np.array(cap)
        #     cap2 = cap/sum(cap)
        #     self.car_1.theta_probability = cap2
        #
        #
        # if self.P.CAR_2.INTENT == 1:
        #     cap = [bij_1,bij_2]
        #     cap = np.array(cap)
        #     cap2 = cap/sum(cap)
        #     self.car_2.theta_probability = cap2
        # else:
        #     cap = [bij_3, bij_4]
        #     cap = np.array(cap)
        #     cap2 = cap/sum(cap)
        #     self.car_2.theta_probability = cap2


        self.trial = self.trial+1
        return np.array(self.state)
