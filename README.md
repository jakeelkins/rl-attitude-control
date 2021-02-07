# rl-attitude-control
code I used for my RL-based spacecraft attitude control research project.

An example gif I made in [Blender](https://www.blender.org/) of an agent I trained with these envs and TD3 detumbling a spacecraft:
![Detumble example](detumble_gif.gif)

Conference papers this was used in: 
- [Discrete variant](https://www.researchgate.net/publication/344659958_Autonomous_Spacecraft_Attitude_Control_Using_Deep_Reinforcement_Learning "Autonomous Spacecraft Attitude Control Using Deep Reinforcement Learning")
- [Continuous variant](https://www.researchgate.net/publication/343834157_Adaptive_Continuous_Control_of_Spacecraft_Attitude_Using_Deep_Reinforcement_Learning "Adaptive Continuous Control of Spacecraft Attitude Using Deep Reinforcement Learning")

I need better titles lol

# Description
This repo has everything I used in studying reinforcement learning for spacecraft attitude control. This repo has a custom OpenAI-gym style simulator ready-to-go for your favorite RL implementation. I included the TD3 implementation I used, along with a quick stable-baselines training notebook to get anyone going.

Files included:
- envs/ADCS_gym_cont (continuous control gym env)
- envs/ADCS_gym_disc (discrete control gym env)
- stable_baselines_trainer.ipynb (training notebook to use with either gym env)
- train_notebook_td3.ipynb (TD3 training notebook I used. only for continuous control)

Feel free to edit the reward functions and all that inside the gym envs themselves to whatever, rotational inertia matrices, etc. 

# Installation
I would just clone down the repo:
> git clone https://github.com/jakeelkins/rl-attitude-control.git

then use the envs and notebooks as you see fit. Feel free to send me a question, my email is on those conference papers.

# Usage
See the two IPython notebooks I included on how to use the envs.

# Packages required
I didn't go thru and make a requirements.txt, so just install whatever part you're using that you don't have. In general, if you have numpy, numba, gym, and pytorch, you'll be good. I used tensorboardX for monitoring with pytorch. 
