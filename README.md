<!-- ⚠️ This README has been generated from the file(s) "blueprint.md" ⚠️--><h1 align="center">reinforcement-learning-workspace</h1>
<p align="center">
  <img src="images/logo.png" alt="Logo" width="550" height="auto" />
</p>


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)](#table-of-contents)

## ➤ Table of Contents

* [➤ :pencil: About The Project](#-pencil-about-the-project)
* [➤ :floppy_disk: Key Project File Description](#-floppy_disk-key-project-file-description)
	* [DQN (Deep Q-Networks)](#dqn-deep-q-networks)
	* [DDPG (Deep Deterministic Policy Gradient)](#ddpg-deep-deterministic-policy-gradient)
	* [TD3 (Twin Delayed DDPG)](#td3-twin-delayed-ddpg)
	* [PPO (Proximal Policy Optimization)](#ppo-proximal-policy-optimization)
	* [MADDPG (Multi-Agent DDPG)](#maddpg-multi-agent-ddpg)
	* [MAPPO (Multi-Agent PPO)](#mappo-multi-agent-ppo)
	* [A3C (Asynchronous Advantage Actor-Critic)](#a3c-asynchronous-advantage-actor-critic)
* [➤ :rocket: Dependencies](#-rocket-dependencies)
* [➤ :hammer: Usage](#-hammer-usage)
* [➤ :coffee: Buy me a coffee](#-coffee-buy-me-a-coffee)
* [➤ :scroll: Credits](#-scroll-credits)
* [➤ License](#-license)
* [➤ License](#-license-1)


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)](#pencil-about-the-project)

## ➤ :pencil: About The Project
--------------------------

This repository is my personal collection and demonstration of various deep reinforcement learning (DRL) algorithms, showcasing my grasp and application of advanced concepts in the field. Featured implementations include popular models such as DQN, A3C, DDPG, TD3, PPO, MADDPG, and MAPPO. Each model's directory provides richly commented code, designed to display not just the technical implementation but also my understanding of the strategic underpinnings of each algorithm. While smaller in scope compared to some of my other projects, this repository serves as an essential tool for both personal reference and as a portfolio piece that reflects my proficiency and ongoing engagement with the latest in DRL advancements.


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)](#floppy_disk-key-project-file-description)

## ➤ :floppy_disk: Key Project File Description

### DQN (Deep Q-Networks)
* The [`DQN folder`](code/dqn/) file implements the DQN algorithm. DQN extends Q-learning by using deep neural networks to approximate the Q-value function. The code includes network architecture, experience replay, and the epsilon-greedy strategy for action selection. It is primarily based on the paper by Mnih et al. (2015), which you can read more about [here](https://www.nature.com/articles/nature14236).

### DDPG (Deep Deterministic Policy Gradient)
* The [`DDPG folder`](code/ddpg/) file contains the implementation of DDPG, a policy gradient algorithm that uses a deterministic policy and operates over continuous action spaces. The script manages network updates, policy learning, and the Ornstein-Uhlenbeck process for action exploration. The foundational paper by Lillicrap et al. (2016) is accessible [here](https://arxiv.org/abs/1509.02971).

### TD3 (Twin Delayed DDPG)
* The [`TD3 folder`](code/td3) file is used for the TD3 algorithm, an extension of DDPG that reduces function approximation error by using twin Q-networks and delayed policy updates. This approach is elaborated in the paper by Fujimoto et al. (2018), which can be found [here](https://arxiv.org/abs/1802.09477).

### PPO (Proximal Policy Optimization)
* The [`PPO fodler`](code/ppo) file facilitates the implementation of PPO, which optimizes policy learning by maintaining a balance between exploration and exploitation using a clipped surrogate objective. The algorithm is detailed in the paper by Schulman et al. (2017), available [here](https://arxiv.org/abs/1707.06347).

### MADDPG (Multi-Agent DDPG)
* The [`MADDPG`](code/maddpg) file explores the MADDPG framework, designed for multi-agent environments. It extends DDPG by considering the actions of other agents in the environment, enhancing training stability and performance in cooperative or competitive scenarios. The key concepts are discussed in the paper by Lowe et al. (2017), accessible [here](https://arxiv.org/abs/1706.02275).

### MAPPO (Multi-Agent PPO)
* The [`MAPPO`](code/mappo) file implements MAPPO, adapting the robust single-agent PPO algorithm for multi-agent settings. This file includes adaptations for centralized training with decentralized execution, suitable for complex multi-agent scenarios. The approach is based on findings discussed in various research studies on multi-agent reinforcement learning.

### A3C (Asynchronous Advantage Actor-Critic)
* COMING SOON


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)](#rocket-dependencies)

## ➤ :rocket: Dependencies

  
**To model the algorithms I used the PyTorch framework only**

![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=for-the-badge) ![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=for-the-badge) ![NumPy Badge](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=fff&style=for-the-badge)



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)](#hammer-usage)

## ➤ :hammer: Usage
 
The easiest way to get started with the deep reinforcement learning algorithms in this repository, is to set up a local development environment. Follow these steps to install and run the implementations:


**Clone the repository:**

```
   git clone https://github.com/i1Cps/Reinforcement_Learning_work.git
   cd Reinforcement_Learning_work

```

**Create a virtual environment (optional but recommended):**
```
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
```

**Install the required dependencies:**
```
    pip3 install -r requirements.txt
```

**Run a specific algorithm (example with PPo):**
```
    cd ppo
    python3 main.py
```

**Plot the results**:
```
    cd data
    python3 plot.py
```

**View graphs in ```<algorithm>/data/plots```



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)](#coffee-buy-me-a-coffee)

## ➤ :coffee: Buy me a coffee
Whether you use this project, have learned something from it, or just like it, please consider supporting it by buying me a coffee, so I can dedicate more time on open-source projects like this (҂⌣̀_⌣́)

<a href="https://www.buymeacoffee.com/i1Cps" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-violet.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)](#scroll-credits)

## ➤ :scroll: Credits

Theo Moore-Calters 


[![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/i1Cps) [![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/theo-moore-calters)


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)](#license)

## ➤ License
	
Licensed under [MIT](https://opensource.org/licenses/MIT).




[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)](#license)

## ➤ License
	
Licensed under [MIT](https://opensource.org/licenses/MIT).
