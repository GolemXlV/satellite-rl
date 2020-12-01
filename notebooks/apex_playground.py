#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ray

ray.init(num_cpus=12, num_gpus=1, memory=1024 * 1024 * 1024 * 10, object_store_memory=1024 * 1024 * 1024 * 30,
#          use_pickle=True,
#          temp_dir='/home/projects/satellite_rl',
)


# In[2]:


from ray.tune.registry import register_env
import gym

def choose_env_for(env_config):
    print(env_config)
    print("worker index is {}".format(env_config.worker_index))
    print("testing vector_index {}".format(env_config.vector_index))
    mod = env_config.worker_index
    if env_config.worker_index > 0:
        mod -= 1
    sat_id = mod * 12 + env_config.vector_index
#     env = gym.make("satellite_gym:SatelliteEnv-v1", sat_id=sat_id)
    env = gym.make("satellite_gym:SatelliteEnv-v2", sat_id=sat_id)
    return env

register_env("SatelliteMultiEnv-v2", lambda x: choose_env_for(x))
# register_env("SatelliteMultiEnv-v1", lambda x: choose_env_for(x))


# In[ ]:


import ray.rllib.agents.ddpg.apex as apex
from ray.tune.logger import pretty_print

def on_train_result(info):
    result = info["result"]
    phase = int(result["episode_reward_mean"]) // int(result["episode_len_mean"] - 2)
#     if result["episode_reward_mean"] > 40:
#         phase = 2
#     elif result["episode_reward_mean"] > 20:
#         phase = 1
#     else:
#         phase = 0
#     phase = 2
#     phase = 0
    trainer = info["trainer"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.set_phase(phase)))

config = apex.APEX_DDPG_DEFAULT_CONFIG.copy()
config['model']['use_lstm'] = True
config["model"]["vf_share_layers"] = True

config["model"]["fcnet_hiddens"] = [512, 512]
config["model"]["lstm_cell_size"] = 512
config["actor_hiddens"] = [500, 400, 300]
config["critic_hiddens"] = [500, 400, 300]

#config["optimizer"]["batch_replay"] = True
# config["num_workers"] = 50
config["num_workers"] = 50
config["num_cpus_per_worker"] = .2
config["seed"] = 0
config["eager"] = False


# Discount factor to 0
#config["gamma"] = 0

# 1-Step Q-learning
#config["n_step"] = 1

# Disable exploration
# config["per_worker_exploration"] = False
config["exploration_should_anneal"] = True
config["schedule_max_timesteps"] = 10000000
config["exploration_ou_noise_scale"] = .01
#config["exploration_fraction"] = 0
config["exploration_final_scale"] = 0
# config["exploration_final_eps"] = 0
#config["pure_exploration_steps"] = 0

config["clip_rewards"] = False
#config["use_huber"] = True
config["actor_lr"] = config["critic_lr"] = config["lr"] = 5e-4
#config["tau"] = 1.0 # value_network + 1-tau/tau * target_network
# config["evaluation_interval"] = 5
# config["evaluation_num_episodes"] = 10
#config["target_network_update_freq"] = 50000
#config["buffer_size"] = 4000000
# config["observation_filter"] = "NoFilter"
#config["train_batch_size"] = 1000
#config["sample_batch_size"] = 100
config["callbacks"] = { "on_train_result": on_train_result }
config["num_envs_per_worker"] = 12
config["env_config"]["num_envs_per_worker"] = config["num_envs_per_worker"]

# trainer = apex.ApexDDPGTrainer(config=config, env="satellite_gym:SatelliteEnv-v1")
trainer = apex.ApexDDPGTrainer.with_updates(default_config=config)
#trainer = apex.ApexDDPGTrainer(config=config, env="SatelliteMultiEnv-v2")
# trainer = apex.ApexDDPGTrainer(config=config, env="SatelliteMultiEnv-v1")

#trainer.restore('/home/golemxiv/ray_results/APEX_DDPG/APEX_DDPG_SatelliteMultiEnv-v2_de9d3f1a_2020-02-07_22-22-59vq4tid4v/checkpoint_150/checkpoint-150')


from ray import tune
#from ray.tune.schedulers import AsyncHyperBandScheduler as ASHA


#scheduler = ASHA(metric="episode_reward_mean", mode="max", max_t=200)

tune.run(
   trainer,
   stop={
#         "training_iteration": 200,
         "episode_reward_mean": 100
   },
   config={
      "env": "SatelliteMultiEnv-v2",
#      "actor_lr": tune.grid_search([1e-5, 3e-3, 3e-4, 3e-5]),
#      "critic_lr": tune.grid_search([1e-5, 3e-3, 3e-4, 3e-5]),
#      "tau": tune.grid_search([0., 2e-2, 1e-3, 1]),
#      "gamma": tune.grid_search([0, .9, .95, .99]),
#      "n_step": tune.grid_search([1, 3, 5]),
      "exploration_ou_noise_scale": .1,
   },
#   scheduler=scheduler,
   checkpoint_freq=50,
   checkpoint_at_end=True,
#   restore='~/ray_results/APEX_DDPG/APEX_DDPG_SatelliteMultiEnv-v2_b004fdca_2020-02-09_00-57-59vd6ws4ss/checkpoint_150/checkpoint-150'
)



#i = 0
#while True:
    # Perform one iteration of training the policy with PPO
#    result = trainer.train()
#    print(pretty_print(result))

#    if i % 50 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)
#    i += 1


# In[ ]:


#trainer.get_policy().export_model('trained_model_v1')
