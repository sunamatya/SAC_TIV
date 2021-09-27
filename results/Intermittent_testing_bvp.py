import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append("C:\\Users\\samatya.ASURITE\\PycharmProjects\\Game-Theoretic-Belief-Distribution")

import gym
from environments.Intent_Inference_BVP_Testing import Intent_Inference_Env
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
import torch


config = Config()
config.seed = 1
config.randomise_random_seed = False
config.environment = Intent_Inference_Env()
config.num_episodes_to_run = 50
config.file_to_save_data_results = "data_and_graphs/Emp_Intent_Inf_Results_Data.pkl"
config.file_to_save_results_graph = "data_and_graphs/Emp_Intent_Inf_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.save_model = False


config.hyperparameters = {
    "Actor_Critic_Agents":  {

        "learning_rate": 0.01, #0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.01,#0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.01,#0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 100,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

if __name__ == "__main__":
    # #the values from below this are from testing
    # the_model = SAC_Discrete(config)
    # #print(the_model.actor_local.state_dict())
    # the_model.actor_local.load_state_dict(torch.load("actor-local"))
    # the_model.critic_local.load_state_dict(torch.load("critic-local"))
    # the_model.critic_local_2.load_state_dict(torch.load("critic-local-2"))
    # #print(the_model.actor_local.state_dict())
    # #print(the_model.actor_optimizer.state_dict())
    # the_model.actor_local.eval()
    # for param in the_model.actor_local.parameters():
    #     print(param)
    # the_model.critic_local.eval()
    # the_model.critic_local_2.eval()
    AGENTS = [SAC_Discrete]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents(eval=True)




