# When Shall I Estimate Your Intent?
## Costs and Benefits of Intent Inference in Multi-Agent Interactions

This repository contains the code for our paper [When Shall I Estimate Your Intent?
Costs and Benefits of Intent Inference in Multi-Agent Interactions](https://ieeexplore.ieee.org/abstract/document/9867155) (ACC-22).

## Table of Contents
1. [Introduction](#introruction)
2. [Motivation](#result_1)
3. [Performance Comparison of Intermittent Empathetic Intent Inference with Baseline Algorithms](#results_2)
4. [Intermittent Intent Inference with Measurement Noise](#results_3)
5. [Instruction to reproduce the results](#instruction)
6. [Acknowledgement](#ackowledgements)
7. [Contact](#contact)
   
## Introduction <a name="introduction"></a>

This paper addresses incomplete-information dynamic games, where reward parameters of agents are private. Previous studies have shown that online belief update is necessary for deriving equilibrial policies of such games, especially for high-risk games such as vehicle interactions. However, updating beliefs in real time is computationally expensive as it requires continuous computation of Nash equilibria of the sub-games starting from the current states. In this paper, we consider the triggering mechanism of belief update as a policy defined on the agents’ physical and belief states, and propose learning this policy through reinforcement learning (RL). Using a two-vehicle uncontrolled intersection case, we show that intermittent belief update via RL is sufficient for safe interactions, reducing the computation cost of updates by 59% when agents have full observations of physical states. Simulation results also show that the belief update frequency will increase as noise becomes more significant in measurements of the vehicle positions.

<img src="paper_results/Slide2.PNG" alt="Drawing" style="height: 200px;"/> 

## Motivation <a name="results_1"></a>
We first study the impact of different intent inference frequencies by letting the AV perform intent inference once in every three time steps. Here, the AV was non-aggressive and the H was aggressive. We perform two simulations with both the AV and H as (1) empathetic, and (2) non-empathetic. In this test, we set the initial positions of both vehicles 2 meters away from the intersection. 

<img src="paper_results/Slide3.PNG" alt="Drawing" style="height: 200px;"/> 

Observation: The empathetic agent chooses to decelerate while the non-empathetic agent chooses to maintain its velocity. Empathetic intent inference with a reduced update frequency shows AV taking significant time to predict the true intent of H. Non-empathetic intent inference with a reduced update frequency shows AV being unable to predict the true intent of the H. This provides preliminary evidence to use empathetic intent inference intermittently. We can see that by just choosing to increase the interval between the belief updates, one can get the correct inference of H’s intent but it takes more instances of belief updates. This motivated us to model the intent inference algorithm as an RL problem, where, instead of reducing the interval between belief update, we let the RL agent decide whether or not to perform belief update at a given time step.

<img src="paper_results/Slide1.PNG" alt="Drawing" style="height: 200px;"/> 
Block diagram of the RL-based intermittent empathetic intent inference. The RL agent takes the states from the environment and provides \textbf{ACTION [0,1]}. The vehicle in the green box represents H, which gives a constant signal $1$ to do belief update at every timestep. Current physical states, $S_p$, input states, $U_p$, and the current joint probability matrix, $B_{\theta}$, are input for the intent inference block, which returns the distribution of the probability of the other agent and the marginal probability of the trajectory. The reactive motion provides action for the AV, which incorporates the inferred motion and intent of the agents.


## Performance Comparison of Intermittent Empathetic Intent Inference with Baseline Algorithms <a name="results_2"></a>
We first pre-train the model where the initial positions, $x_e$ and $x_{o}$, are randomly sampled from $x_e \in [1,2,3]$ for the $x$-coordinate, where $o$ and $e$ stand for the pursuer/other and evader/ego in the game, respectively. The $y$-coordinate fixed at 0.  This pre-training is done on Task 1, where the goal position is set to (2,5). After pre-training, we retrain the model on new tasks with changing goal positions: Task 2 has a goal at (2, 4), Task 3 at (2, 1), and Task 4 at (2, 0).
In this work, we explore how SFs with GGPI facilitates transfer to new goal positions in multi agent games. During training, we evaluate the performance at the midpoint of the game and examine the results after final convergence.
<img src="paper_results/Slide4.PNG" alt="Drawing" style="height: 200px;"/> 

Observation: a) Possible initial position and goals for testing transfer with significant reward change (b) Path taken by SFminmax agents where pursuer agent
is able to intervene. (c) Path taken my Minmax agents where pursuer is not to intervene (d) Cumulative return per task in the Pursuer-Evader game.
SFminmax reward uses the feature from the previous task at the beginning to each task.

## [Intermittent Intent Inference with Measurement Noise] <a name="results_3"></a>
We first pre-train the model where the initial positions, $x_e$ and $x_{o}$, are randomly sampled from $x_e \in [1,2,3]$ for the $x$-coordinate, where $o$ and $e$ stand for the pursuer/other and evader/ego in the game, respectively. The $y$-coordinate fixed at 0.  This pre-training is done on Task 1, where the goal position is set to (2,5). After pre-training, we retrain the model on new tasks with changing goal positions: Task 2 has a goal at (2, 4), Task 3 at (2, 1), and Task 4 at (2, 0).
In this work, we explore how SFs with GGPI facilitates transfer to new goal positions in multi agent games. During training, we evaluate the performance at the midpoint of the game and examine the results after final convergence.
<img src="paper_results/Slide4.PNG" alt="Drawing" style="height: 200px;"/> 

Observation: a) Possible initial position and goals for testing transfer with significant reward change (b) Path taken by SFminmax agents where pursuer agent
is able to intervene. (c) Path taken my Minmax agents where pursuer is not to intervene (d) Cumulative return per task in the Pursuer-Evader game.
SFminmax reward uses the feature from the previous task at the beginning to each task.


## Instructions to reproduce the results <a name="instruction"></a>
As a classic reinforcement learning algorithm, we need to look into the environment and the agent.
To run the experiment, we use results\Intermittent_Emp_Int_Inf.py. Here the environment, actors and configuration for the reinforcement learning problem is defined as well as the configuration to store the graphs and data. The hyperparameters for the learning agnet is also defined in this file. For the second experiments on the noise level $config.environment.noise\_level$ is used. The noise level is thus used in the Environment. The Environment is defined is defined in gym. The training environment of the RL problem is in environments\Intent_Inference_Env_Training. This converts the original [SocialGracefulness](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8911426) into a gym environment for the RL problem. 

## Acknowledgement  <a name="acknowledgement"></a>
This repo contains code that's based on the following code: [Deep Reinforcement Learning Algorithms with PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch).
This repo requires working in conjunction with the following code base: [SocialGracefullnessTIV](https://github.com/sunamatya/SocialGracefullnessTIV)


## BibTeX

If you find this repository or the ideas presented in our paper useful for your research, please consider citing our paper.

@inproceedings{amatya2022shall,
  title={When Shall I Estimate Your Intent? Costs and Benefits of Intent Inference in Multi-Agent Interactions},
  author={Amatya, Sunny and Ghimire, Mukesh and Ren, Yi and Xu, Zhe and Zhang, Wenlong},
  booktitle={2022 American Control Conference (ACC)},
  pages={586--592},
  year={2022},
  organization={IEEE}
}


## Contact <a name="contact"></a>

Feel free to contact <a href="mailto:Wenlong.Zhang@asu.edu">Wenlong Zhang</a> or <a href="mailto:sunnyamatya@gmail.com">Sunny Amatya</a> for any questions regarding the code or the paper. Please visit our website for more information: [RISE website](https://home.riselab.info/).
