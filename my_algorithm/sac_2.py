#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from parl.core.fluid import layers
from copy import deepcopy
import numpy as np
import paddle
from paddle import fluid
from paddle.fluid.layers import Normal
from parl.core.fluid.algorithm import Algorithm


# fluid.enable_dygraph()
# fluid.dygraph.guard()


# import paddle

# from paddle.distribution import Normal
# from visualdl import LogWriter
# import random
# import collections
# import gym
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from tqdm import tqdm
# import numpy as np
# import copy


epsilon = 1e-6

REWARD_alpha = 0.02


REWARD_LR = 0.05

EXP_FAST_LEARNING_RATE = 1e-3


# code intrinsic_reward
IR_STEPS = 3
IR_RATE = 10


__all__ = ['SAC_2']


class SAC_2(Algorithm):
    def __init__(self,
                 actor,
                 critic,
                 max_action,
                 alpha=0.2,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None,
                 alpha_lr=None,
                 alpha_model=None,
                 reward_model=None,
                 batch_size=100,
                 states_model=None,
                 states_lr=0.005,
                 is_double=False,
                 ):
        """ SAC algorithm

        Args:
            actor (parl.Model): forward network of actor.
            critic (patl.Model): forward network of the critic.
            max_action (float): the largest value that an action can be, env.action_space.high[0]
            alpha (float): Temperature parameter determines the relative importance of the entropy against the reward
            gamma (float): discounted factor for reward computation.
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            actor_lr (float): learning rate of the actor model
            critic_lr (float): learning rate of the critic model
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        assert isinstance(alpha_lr, float)
        assert isinstance(alpha, float)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.target_entropy = -max_action
        self.batch_size = batch_size
        self.max_delta_action = 0.1
        self.is_double=is_double

        # external_alpha_value = paddle.to_tensor(alpha, dtype='float32')

        # 使用Constant初始化器，值来自external_alpha_value
        # with fluid.dygraph.guard():
        #     self.alpha = fluid.dygraph.to_variable(
        #         # shape=[1],
        #         # dtype='float32',
        #         # default_initializer=fluid.initializer.Constant(value=np.log(alpha))
        #         value = np.array([alpha])
        #     )

        # with fluid.dygraph.guard():
        #     self.alpha = fluid.layers.create_parameter(
        #         shape=[1],
        #         dtype='float32',
        #         default_initializer=fluid.initializer.Constant(value=np.log(alpha)),
        #         name = "alpha"
        #         # value = np.array([alpha])
        #     )
        # self.alpha.stop_gradient = False
        self.alpha = alpha

        self.actor = actor
        self.critic = critic
        self.alpha_model = alpha_model
        self.reward_model = reward_model
        self.target_critic = deepcopy(critic)

    # def predict(self, obs, last_action):
    #     """ use actor model of self.policy to predict the action
    #     """
        
    #     with fluid.dygraph.guard():
    #         print("obs shape = ", obs.shape)
    #         print("last_action shape = ", last_action.shape)
            
        
    #     mean, _ = self.actor.policy(layers.concat(input=[obs, last_action], axis=-1))
    #     mean = layers.tanh(mean) * self.max_action
        
    #     # mean = layers.elementwise_add(self.max_delta_action * mean, last_action)
        
    #     # mean = layers.clip(mean, min=(-1.0) *self.max_action, max=(1.0) *self.max_action)
    #     mean = layers.clip(mean, min=-1.0, max=1.0)
        
    #     return mean
    
    
    
    def predict(self, obs, last_action):
        """ use actor model of self.policy to predict the action
        """
        
        with fluid.dygraph.guard():
            print("obs shape = ", obs.shape)
            print("last_action shape = ", last_action.shape)
            
        # obs = batch_normalize(obs)
            
        
        mean, pred_trend, pred_amplitude, log_std = self.actor.policy(layers.concat(input=[obs, last_action], axis=-1))
        
        if self.is_double:
            mean = pred_trend
        # mean = layers.tanh(mean) * self.max_action
        
        # mean = layers.elementwise_add(self.max_delta_action * mean, last_action)
        
        # mean = layers.clip(mean, min=(-1.0) *self.max_action, max=(1.0) *self.max_action)
        mean = layers.clip(mean, min=-1.0, max=1.0)
        
        return mean

    # def sample(self, obs, last_action):
    #     mean, log_std = self.actor.policy(layers.concat(input=[obs, last_action], axis=-1))
    #     std = layers.exp(log_std)
    #     normal = Normal(mean, std)
    #     x_t = normal.sample([1])[0]
    #     y_t = layers.tanh(x_t)
    #     action = y_t * self.max_action
    #     log_prob = normal.log_prob(x_t)
    #     log_prob -= layers.log(self.max_action * (1 - layers.pow(y_t, 2)) +
    #                            epsilon)
    #     log_prob = layers.reduce_sum(log_prob, dim=1, keep_dim=True)
    #     log_prob = layers.squeeze(log_prob, axes=[1])
        
        
    #     # action = layers.elementwise_add(self.max_delta_action * action, last_action)
        
    #     # action = layers.clip(action, min=(-1.0) *self.max_action, max=(1.0) *self.max_action)
        
    #     action = layers.clip(action, min=-1.0, max=1.0)
        
    #     return action, log_prob
    
    
    def sample(self, obs, last_action):
        pred_combined, pred_trend, pred_amplitude, log_std = self.actor.policy(layers.concat(input=[obs, last_action], axis=-1))
        std = layers.exp(log_std)
        normal = Normal(pred_combined, std)
        x_t = normal.sample([1])[0]
        y_t = layers.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= layers.log(self.max_action * (1 - layers.pow(y_t, 2)) +
                               epsilon)
        log_prob = layers.reduce_sum(log_prob, dim=1, keep_dim=True)
        log_prob = layers.squeeze(log_prob, axes=[1])
        
        action = layers.clip(action, min=-1.0, max=1.0)
        
        return action, log_prob
    
    
    
    def sample_double(self, obs, last_action):
        pred_combined, pred_trend, pred_amplitude, log_std = self.actor.policy(layers.concat(input=[obs, last_action], axis=-1))
        std = layers.exp(log_std)
        normal = Normal(pred_trend, std)
        x_t = normal.sample([1])[0]
        y_t = layers.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= layers.log(self.max_action * (1 - layers.pow(y_t, 2)) +
                               epsilon)
        log_prob = layers.reduce_sum(log_prob, dim=1, keep_dim=True)
        log_prob = layers.squeeze(log_prob, axes=[1])
        
        action = layers.clip(action, min=-1.0, max=1.0)
        
        return action, log_prob
    
    def get_pred_trend(self, obs, last_action):
        # mean, log_std = self.actor.policy(layers.concat(input=[obs, last_action], axis=-1))
        # action = layers.tanh(mean) * self.max_action
        
        action, trend = self.actor.trend_policy(layers.concat(input=[obs, last_action], axis=-1))
        # action = layers.tanh(action/1e-3)
        
        # layers.Print(self.actor.pa)
        
        # action_dif = layers.elementwise_add(action, -last_action)
        # action_dif_sign = layers.tanh(action_dif/1e-3)
        # return action_dif_sign, action
        return action, trend
    
    def get_only_trend(self, obs, last_action):
        # obs = batch_normalize(obs)
        combined_s = layers.concat(input=[obs, last_action], axis=-1)
        with fluid.dygraph.guard():
            print("[DOUBLE NETWORK DEBUG] obs shape = ", obs.shape)
            print("[DOUBLE NETWORK DEBUG] last_action shape = ", last_action.shape)
            print("[DOUBLE NETWORK DEBUG] combined_s shape = ", combined_s.shape)
        combined_s = layers.reshape(combined_s, [-1, combined_s.shape[1]])
        with fluid.dygraph.guard():
            print("[DOUBLE NETWORK DEBUG] combined_s 2 shape = ", combined_s.shape)
        action, trend = self.actor.trend_policy(combined_s)
        return trend
    
    def get_only_ampl(self, obs, last_action):
        # obs = batch_normalize(obs)
        amplitude, log_std = self.actor.amplitude_policy(layers.concat(input=[obs, last_action], axis=-1))
        return amplitude

    def sample_reward(self, obs, action):

        # 假设奖励的均值和标准差由某个策略网络输出
        mean_reward, log_std_reward = self.reward_model.forward(
            layers.concat(input=[obs, action], axis=-1))
        std_reward = layers.exp(log_std_reward)

        # 创建正态分布用于生成奖励
        reward_distribution = Normal(mean_reward, std_reward)
        reward = reward_distribution.sample([1])[0]  # 采样奖励

        # 计算奖励的对数概率
        log_prob_reward = reward_distribution.log_prob(reward)
        # # fluid.layers.Print(log_prob_reward)
        log_prob_reward = layers.reduce_sum(
            log_prob_reward, dim=1, keep_dim=True)
        # # fluid.layers.Print(log_prob_reward)
        log_prob_reward = layers.squeeze(log_prob_reward, axes=[1])
        # fluid.layers.Print(log_prob_reward)

        prob_reward = layers.exp(log_prob_reward/100.0)
        entropy_reward = layers.elementwise_mul(prob_reward, log_prob_reward)
        # fluid.layers.Print(prob_reward)
        # fluid.layers.Print(entropy_reward)
        # 奖励熵的计算可以通过 -log_prob_reward 来近似
        entropy_reward = -layers.reduce_mean(log_prob_reward)

        return reward, entropy_reward, mean_reward, log_std_reward

    def learn(self, obs, action, reward, next_obs, terminal, last_action):
        # print("[[[[[[[[[[[[[[[[[[[[[[[[[[[[ enter learn ]]]]]]]]]]]]]]]]]]]]]]]]]]]]")


        actor_cost, alpha_cost = self.actor_and_alpha_learn(obs, reward, last_action)
        critic_cost = self.critic_learn(obs, action, reward, next_obs,
                                        terminal, last_action)

        # # test
        # # Calculate histogram
        # hist, bin_edges = custom_histogram(np.array(reward), bins=100)
        # total_counts = np.sum(hist)

        # alpha_cost = self.alpha_learn(xxxxxxxxxxxxx)
        return critic_cost, actor_cost, alpha_cost

    # def actor_and_alpha_learn(self, obs, reward, last_action):
    #     self.alpha = self.alpha_model.forward(obs)

    #     # # fluid.layers.Print(self.alpha)

    #     # fluid.layers.Print(obs)
    #     # fluid.layers.Print(reward)

    #     action, log_pi = self.sample(obs, last_action)
    #     # action, log_pi = self.sample(layers.concat(input=[obs, reward], axis=0))
    #     #action, log_pi = self.sample(layers.concat(input=[obs, layers.unsqueeze(reward, axes=1)], axis=-1))

    #     qf1_pi, qf2_pi = self.critic.value(obs, action)
    #     min_qf_pi = layers.elementwise_min(qf1_pi, qf2_pi)

    #     actor_cost = log_pi * self.alpha - min_qf_pi
    #     actor_cost = layers.reduce_mean(actor_cost)

    #     # fluid.layers.Print(actor_cost)

    #     # actor_cost = layers.reduce_mean(actor_cost)
    #     actor_optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
    #     actor_optimizer.minimize(
    #         actor_cost, parameter_list=self.actor.parameters())

    #     alpha_loss = (-log_pi - self.target_entropy) * self.alpha
    #     alpha_loss = layers.reduce_mean(alpha_loss)
    #     alpha_optimizer = fluid.optimizer.AdamOptimizer(self.alpha_lr)
    #     alpha_optimizer.minimize(
    #         alpha_loss, parameter_list=self.alpha_model.parameters())

    #     return actor_cost, alpha_loss



    def actor_and_alpha_learn(self, obs, reward, last_action):
        self.alpha = self.alpha_model.forward(obs)

        if self.is_double:
            action, log_pi = self.sample_double(obs, last_action)
        else:
            action, log_pi = self.sample(obs, last_action)
        # action, log_pi = self.sample(layers.concat(input=[obs, reward], axis=0))
        #action, log_pi = self.sample(layers.concat(input=[obs, layers.unsqueeze(reward, axes=1)], axis=-1))

        qf1_pi, qf2_pi = self.critic.value(obs, action)
        min_qf_pi = layers.elementwise_min(qf1_pi, qf2_pi)

        actor_cost = log_pi * self.alpha - min_qf_pi
        actor_cost = layers.reduce_mean(actor_cost)

        # fluid.layers.Print(actor_cost)

        # actor_cost = layers.reduce_mean(actor_cost)
        actor_optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        actor_optimizer.minimize(
            actor_cost, parameter_list=self.actor.parameters())

        alpha_loss = (-log_pi - self.target_entropy) * self.alpha
        alpha_loss = layers.reduce_mean(alpha_loss)
        alpha_optimizer = fluid.optimizer.AdamOptimizer(self.alpha_lr)
        alpha_optimizer.minimize(
            alpha_loss, parameter_list=self.alpha_model.parameters())

        return actor_cost, alpha_loss





    def compute_loss(pred_combined, pred_trend, pred_amplitude, target_combined, target_trend):
        # 趋势预测的交叉熵损失
        target_trend_mapped = layers.elementwise_add(target_trend, layers.fill_constant(shape=target_trend.shape, value=1.0))
        target_trend_mapped = layers.cast(target_trend_mapped, dtype='int64')
        trend_loss = layers.reduce_mean(layers.softmax_with_cross_entropy(pred_trend, target_trend_mapped))

        # 幅度预测的 MSE 损失
        amplitude_loss = layers.reduce_mean(layers.square(pred_amplitude - layers.abs(target_combined)))

        # 总损失
        total_loss = trend_loss + 0.5 * amplitude_loss  # 0.5 是幅度损失的权重
        return total_loss





    def actor_learn(self, obs):
        action, log_pi = self.sample(obs)
        qf1_pi, qf2_pi = self.critic.value(obs, action)
        min_qf_pi = layers.elementwise_min(qf1_pi, qf2_pi)
        cost = log_pi * self.alpha - min_qf_pi
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        # optimizer.minimize(cost, parameter_list=self.actor.parameters())

        return cost
    
    # # TODO
    # def exp_fast_learn(self, obs, last_action, exp_trend, A):
    #     # p_action, log_pi = self.sample(obs, last_action)
    #     # exp_fast_loss = layers.elementwise_add(exp_trend, -p_action)
    #     # exp_fast_loss = layers.elementwise_pow(exp_fast_loss, 2)
    #     # exp_fast_loss = layers.square(exp_fast_loss)
        
    #     # pred_action_trend, action = self.get_pred_trend(obs, last_action)
        
        
        
    #     pred_action_trend = self.get_pred_trend(obs, last_action)
        
    #     with fluid.dygraph.guard():
    #         print("[PD MERGE DEBUG] pred_action_trend.shape = ", pred_action_trend.shape)
    #         print("[PD MERGE DEBUG] last_action.shape = ", last_action.shape)
    #         print("[PD MERGE DEBUG] exp_trend.shape = ", exp_trend.shape)
    #         print("[PD MERGE DEBUG] A.shape = ", A.shape)
        
    #     # 2D--->3D
    #     pred_action_trend_bmm = layers.unsqueeze(pred_action_trend, axes=[-1])
    #     with fluid.dygraph.guard():
    #         print("[PD MERGE DEBUG] pred_action_trend_bmm.shape = ", pred_action_trend_bmm.shape)
    #     pred_action_trend = layers.bmm(A, pred_action_trend_bmm)
    #     # 3D--->2D
    #     pred_action_trend = layers.squeeze(pred_action_trend, axes=[-1])
    #     with fluid.dygraph.guard():
    #         print("[PD MERGE DEBUG] pred_action_trend2.shape = ", pred_action_trend.shape)
        
    #     exp_fast_loss = layers.elementwise_add(pred_action_trend, -exp_trend)
    #     exp_fast_loss = layers.square(exp_fast_loss)
        
    #     loss_continuous = exp_fast_loss
        
    #     # 计算离散化误差 (离散值与目标值的误差)
    #     rounded_pred = layers.round(pred_action_trend)
    #     loss_discrete = layers.square(rounded_pred - exp_trend)

    #     # 加权总损失
    #     lambda_coeff = 0.05  # 离散化损失的权重
    #     exp_fast_loss = layers.reduce_mean(loss_continuous) + lambda_coeff * layers.reduce_mean(loss_discrete)
        
        
        
    #     with fluid.dygraph.guard():
    #         print("[PD MERGE DEBUG] pred_action_trend.shape = ", pred_action_trend.shape)
        
    #     # mean_action = layers.reduce_mean(action, dim=0, keep_dim=True)
    #     # exp_fast_loss = layers.reduce_mean(exp_fast_loss) - 1.5 * layers.reduce_mean(layers.square(action - mean_action))
    #     exp_fast_loss = layers.reduce_mean(exp_fast_loss)
    #     # 0.5 * layers.reduce_mean(layers.square(action - mean_action))
        
    #     exp_fast_optimizer = fluid.optimizer.AdamOptimizer(EXP_FAST_LEARNING_RATE)
    #     exp_fast_optimizer.minimize(exp_fast_loss, parameter_list=self.actor.parameters())
        
    #     return exp_fast_loss
    
    
    # TODO
    def exp_fast_learn(self, obs, last_action, exp_trend, A):
        if self.is_double:
            # return 0.00
            return layers.fill_constant(shape=[1], dtype='float32', value=-1.00)

        
        pred_action_trend, single_trend = self.get_pred_trend(obs, last_action)
        
        with fluid.dygraph.guard():
            print("[PD MERGE DEBUG] pred_action_trend.shape = ", pred_action_trend.shape)
            print("[PD MERGE DEBUG] last_action.shape = ", last_action.shape)
            print("[PD MERGE DEBUG] exp_trend.shape = ", exp_trend.shape)
            print("[PD MERGE DEBUG] A.shape = ", A.shape)
            # print("[PD MERGE DEBUG] pred_action_trend = ", pred_action_trend.numpy())
            
            
        
        # 2D--->3D
        pred_action_trend_bmm = layers.unsqueeze(pred_action_trend, axes=[-1])
        pred_action_trend_bmm = layers.squeeze(pred_action_trend_bmm, axes=[-1])
        with fluid.dygraph.guard():
            print("[PD MERGE DEBUG] pred_action_trend_bmm.shape = ", pred_action_trend_bmm.shape)
        pred_action_trend = layers.bmm(A, pred_action_trend_bmm)
        # 3D--->2D
        # pred_action_trend = layers.squeeze(pred_action_trend, axes=[-1])
        with fluid.dygraph.guard():
            print("[PD MERGE DEBUG] pred_action_trend2.shape = ", pred_action_trend.shape)
        
         # 趋势预测的交叉熵损失
        # target_trend_mapped = layers.elementwise_add(pred_action_trend, layers.fill_constant(shape=pred_action_trend.shape, value=1.0, dtype='float32'))
        # target_trend_mapped = layers.cast(target_trend_mapped, dtype='int64')
        
        # target_trend_mapped = pred_action_trend
        # target_trend_mapped = layers.reshape(target_trend_mapped, shape=[-1])
        # exp_trend = layers.reshape(exp_trend, shape=[-1])
        
        # with fluid.dygraph.guard():
        #     print("[PD MERGE DEBUG] target_trend_mapped.shape = ", target_trend_mapped.shape)
        #     print("[PD MERGE DEBUG] exp_trend.shape = ", exp_trend.shape)
            
        # exp_fast_loss = layers.reduce_mean(layers.softmax_with_cross_entropy(exp_trend, target_trend_mapped, soft_label=False))
        exp_trend = fluid.layers.reshape(exp_trend, shape=[-1, 1])
        exp_trend = fluid.layers.cast(exp_trend, dtype='int64')
        pred_action_trend = fluid.layers.reshape(pred_action_trend, shape=[-1, 3])  # 调整为 [batch_size * action_dim, 3]
        # 损失函数：交叉熵
        exp_fast_loss = fluid.layers.softmax_with_cross_entropy(logits=pred_action_trend, label=exp_trend, soft_label=False)
        exp_fast_loss = fluid.layers.reduce_mean(exp_fast_loss)  # 平均损失
        
        # exp_fast_loss = layers.elementwise_add(pred_action_trend, -exp_trend)
        # exp_fast_loss = layers.square(exp_fast_loss)
        
        
        with fluid.dygraph.guard():
            print("[PD MERGE DEBUG] pred_action_trend.shape = ", pred_action_trend.shape)
        
        # mean_action = layers.reduce_mean(action, dim=0, keep_dim=True)
        # exp_fast_loss = layers.reduce_mean(exp_fast_loss) - 1.5 * layers.reduce_mean(layers.square(action - mean_action))
        exp_fast_loss = layers.reduce_mean(exp_fast_loss)
        # 0.5 * layers.reduce_mean(layers.square(action - mean_action))
        grad_clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0)
        exp_fast_optimizer = fluid.optimizer.AdamOptimizer(EXP_FAST_LEARNING_RATE, grad_clip=grad_clip)
        # exp_fast_optimizer = fluid.optimizer.AdamOptimizer(EXP_FAST_LEARNING_RATE)
        exp_fast_optimizer.minimize(exp_fast_loss, parameter_list=self.actor.parameters())
        
        return exp_fast_loss
       
       
    # TODO: 将双模型改造成输出连续值
    def exp_fast_learn_double(self, obs, last_action, exp_trend, A):
        if not self.is_double:
            # return 0.00
            return layers.fill_constant(shape=[1], dtype='float32', value=-1.00)

        
        pred_action_trend, single_trend = self.get_pred_trend(obs, last_action)
        pred_action_trend = single_trend
        
        with fluid.dygraph.guard():
            print("[PD MERGE DEBUG] pred_action_trend.shape = ", pred_action_trend.shape)
            print("[PD MERGE DEBUG] last_action.shape = ", last_action.shape)
            print("[PD MERGE DEBUG] exp_trend.shape = ", exp_trend.shape)
            print("[PD MERGE DEBUG] A.shape = ", A.shape)
            # print("[PD MERGE DEBUG] pred_action_trend = ", pred_action_trend.numpy())
            
            
        
        # 2D--->3D
        pred_action_trend_bmm = layers.unsqueeze(pred_action_trend, axes=-1)
        # pred_action_trend_bmm = layers.unsqueeze(pred_action_trend_bmm, axes=-1)
        pred_action_trend_bmm = layers.reshape(pred_action_trend, shape=[-1, pred_action_trend.shape[1], 1])
        # 假设 pred_action_trend_bmm 的形状是 (-1, 10)
        # pred_action_trend_bmm = paddle.unsqueeze(pred_action_trend_bmm, axis=-1)  # 扩展到 (-1, 10, 1)
        # pred_action_trend_bmm = layers.squeeze(pred_action_trend_bmm, axes=[-1])
        with fluid.dygraph.guard():
            print("[PD MERGE DEBUG] pred_action_trend_bmm.shape = ", pred_action_trend_bmm.shape)
        pred_action_trend = layers.bmm(A, pred_action_trend_bmm)
        # 3D--->2D
        pred_action_trend = layers.squeeze(pred_action_trend, axes=[-1])
        with fluid.dygraph.guard():
            print("[PD MERGE DEBUG] pred_action_trend2.shape = ", pred_action_trend.shape)
        
        
        # exp_trend = fluid.layers.reshape(exp_trend, shape=[-1, 1])
        exp_trend = fluid.layers.cast(exp_trend, dtype='float32')
        # pred_action_trend = fluid.layers.reshape(pred_action_trend, shape=[-1, 3])  # 调整为 [batch_size * action_dim, 3]
        # # 损失函数：交叉熵
        # exp_fast_loss = fluid.layers.softmax_with_cross_entropy(logits=pred_action_trend, label=exp_trend, soft_label=False)
        # exp_fast_loss = fluid.layers.reduce_mean(exp_fast_loss)  # 平均损失
        
        exp_fast_loss = layers.elementwise_add(pred_action_trend, -exp_trend)
        exp_fast_loss = layers.square(exp_fast_loss)
        
        
        with fluid.dygraph.guard():
            print("[PD MERGE DEBUG] pred_action_trend.shape = ", pred_action_trend.shape)
        
        # mean_action = layers.reduce_mean(action, dim=0, keep_dim=True)
        # exp_fast_loss = layers.reduce_mean(exp_fast_loss) - 1.5 * layers.reduce_mean(layers.square(action - mean_action))
        exp_fast_loss = layers.reduce_mean(exp_fast_loss)
        # 0.5 * layers.reduce_mean(layers.square(action - mean_action))
        grad_clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0)
        exp_fast_optimizer = fluid.optimizer.AdamOptimizer(EXP_FAST_LEARNING_RATE, grad_clip=grad_clip)
        # exp_fast_optimizer = fluid.optimizer.AdamOptimizer(EXP_FAST_LEARNING_RATE)
        exp_fast_optimizer.minimize(exp_fast_loss, parameter_list=self.actor.parameters())
        
        return exp_fast_loss
        

    def critic_learn(self, obs, action, reward, next_obs, terminal, last_action):
        next_obs_action, next_obs_log_pi = self.sample(next_obs, last_action)
        qf1_next_target, qf2_next_target = self.target_critic.value(
            next_obs, next_obs_action)
        min_qf_next_target = layers.elementwise_min(
            qf1_next_target, qf2_next_target) - next_obs_log_pi * self.alpha

        terminal = layers.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.gamma * min_qf_next_target
        target_Q.stop_gradient = True

        current_Q1, current_Q2 = self.critic.value(obs, action)
        # # fluid.layers.Print(current_Q1)
        cost = layers.square_error_cost(current_Q1,
                                        target_Q) + layers.square_error_cost(
                                            current_Q2, target_Q)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(self.critic_lr)
        optimizer.minimize(cost)
        return cost

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.critic.sync_weights_to(self.target_critic, decay=decay)

    def cal_td_error(self, obs, action, reward, next_obs, terminal):
        print("******************************************")
        print("**                                      **")
        print("**       this is sac cal_td_error       **")
        print("**      has not been implemented!!      **")
        print("**                                      **")
        print("******************************************")
        return 0.0


def custom_histogram(rewards, bins=10):
    min_val = np.min(rewards)
    max_val = np.max(rewards)
    range = max_val - min_val
    bin_edges = np.linspace(min_val, max_val, bins+1)
    bin_counts = [0] * bins

    for reward in rewards:
        if reward == max_val:
            bin_counts[-1] += 1
        else:
            index = int((reward - min_val) / range * bins)
            bin_counts[index] += 1

    return np.array(bin_counts), bin_edges


def calculate_entropy_from_histogram(hist, total):
    probabilities = hist / total
    # Filter zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy



def standardize(obs):
    """
    对单个样本进行标准化处理。

    参数:
        obs (np.ndarray): 单个样本，1D 数组。

    返回:
        paddle.Tensor: 标准化后的样本。
    """
    # 转换为 Paddle Tensor
    obs_tensor = obs

    # 计算均值和标准差
    mean = fluid.layers.reduce_mean(obs_tensor)
    std = fluid.layers.reduce_mean(fluid.layers.sqrt((obs_tensor - mean)**2))

    # 标准化公式
    obs_normalized = (obs_tensor - mean) / (std + 1e-8)
    return obs_normalized


def batch_normalize(obs_array, method='standardize'):
    """
    对多个样本按列归一化处理。

    参数:
        obs_array (np.ndarray): 多个样本，2D 数组。
        method (str): 归一化方法，'standardize' 或 'min_max'。

    返回:
        paddle.Tensor: 归一化后的样本。
    """
    obs_tensor = obs_array

    if method == 'standardize':
        mean = fluid.layers.reduce_mean(obs_tensor, dim=0, keep_dim=True)
        std = fluid.layers.reduce_mean(fluid.layers.sqrt((obs_tensor - mean)**2), dim=0, keep_dim=True)
        obs_normalized = (obs_tensor - mean) / (std + 1e-8)
    elif method == 'min_max':
        min_val = fluid.layers.reduce_min(obs_tensor, dim=0, keep_dim=True)
        max_val = fluid.layers.reduce_max(obs_tensor, dim=0, keep_dim=True)
        obs_normalized = (obs_tensor - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError("Unsupported normalization method: {}".format(method))

    return obs_normalized






    # ''' 处理离散动作的SAC算法 '''
    # def __init___0(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
    #              alpha_lr, target_entropy, tau, gamma):
    #     # 策略网络
    #     self.actor = PolicyNet(state_dim, hidden_dim, action_dim)
    #     # 第一个Q网络
    #     self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim)
    #     # 第二个Q网络
    #     self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim)
    #     self.target_critic_1 = QValueNet(state_dim, hidden_dim,
    #                                      action_dim)  # 第一个目标Q网络
    #     self.target_critic_2 = QValueNet(state_dim, hidden_dim,
    #                                      action_dim)  # 第二个目标Q网络
    #     # 令目标Q网络的初始参数和Q网络一样
    #     self.target_critic_1.set_state_dict(self.critic_1.state_dict())
    #     self.target_critic_2.set_state_dict(self.critic_2.state_dict())

    #     self.actor_optimizer = paddle.optimizer.Adam(parameters = self.actor.parameters(),
    #                                             learning_rate=actor_lr)
    #     self.critic_1_optimizer = paddle.optimizer.Adam(parameters = self.critic_1.parameters(),
    #                                                learning_rate=critic_lr)
    #     self.critic_2_optimizer = paddle.optimizer.Adam(parameters = self.critic_2.parameters(),
    #                                                learning_rate=critic_lr)

    #     # 使用alpha的log值,可以使训练结果比较稳定
    #     self.log_alpha = paddle.to_tensor(np.log(0.01), dtype="float32")
    #     self.log_alpha.stop_gradient  = False  # 可以对alpha求梯度
    #     self.log_alpha_optimizer = paddle.optimizer.Adam(parameters = [self.log_alpha],
    #                                                 learning_rate=alpha_lr)

    #     self.target_entropy = target_entropy  # 目标熵的大小
    #     self.gamma = gamma
    #     self.tau = tau

    # def save(self):
    #     paddle.save(self.actor.state_dict(),'net.pdparams')

    # def take_action(self, state):
    #     state = paddle.to_tensor([state], dtype="float32")
    #     probs = self.actor(state)
    #     action_dist = paddle.distribution.Categorical(probs)
    #     action = action_dist.sample([1])
    #     return action.numpy()[0][0]

    # # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    # def calc_target(self, rewards, next_states, dones):
    #     next_probs = self.actor(next_states)
    #     next_log_probs = paddle.log(next_probs + 1e-8)
    #     entropy = -paddle.sum(next_probs * next_log_probs, axis=1, keepdim=True)
    #     q1_value = self.target_critic_1(next_states)
    #     q2_value = self.target_critic_2(next_states)
    #     min_qvalue = paddle.sum(next_probs * paddle.minimum(q1_value, q2_value),
    #                            axis=1,
    #                            keepdim=True)
    #     next_value = min_qvalue + self.log_alpha.exp() * entropy
    #     td_target = rewards + self.gamma * next_value * (1 - dones)
    #     return td_target

    # def soft_update(self, net, target_net):
    #     for param_target, param in zip(target_net.parameters(),
    #                                    net.parameters()):
    #         param_target.set_value(param_target * (1.0 - self.tau) + param * self.tau)

    # def update(self, transition_dict):
    #     states = paddle.to_tensor(transition_dict['states'],dtype="float32")
    #     actions = paddle.to_tensor(transition_dict['actions']).reshape([-1, 1])
    #     rewards = paddle.to_tensor(transition_dict['rewards'],dtype="float32").reshape([-1, 1])
    #     next_states = paddle.to_tensor(transition_dict['next_states'],dtype="float32")
    #     dones = paddle.to_tensor(transition_dict['dones'],dtype="float32").reshape([-1, 1])

    #     # 更新两个Q网络
    #     td_target = self.calc_target(rewards, next_states, dones)
    #     critic_1_q_values = paddle_gather(self.critic_1(states), 1, actions)

    #     critic_1_loss = paddle.mean(F.mse_loss(critic_1_q_values, td_target.detach()))

    #     critic_2_q_values = paddle_gather(self.critic_2(states), 1, actions)

    #     critic_2_loss = paddle.mean( F.mse_loss(critic_2_q_values, td_target.detach()))

    #     self.critic_1_optimizer.clear_grad()
    #     critic_1_loss.backward()
    #     self.critic_1_optimizer.step()
    #     self.critic_2_optimizer.clear_grad()
    #     critic_2_loss.backward()
    #     self.critic_2_optimizer.step()

    # # 更新策略网络
    # probs = self.actor(states)
    # log_probs = paddle.log(probs + 1e-8)
    # # 直接根据概率计算熵
    # entropy = -paddle.sum(probs * log_probs, axis=1, keepdim=True)  #
    # q1_value = self.critic_1(states)
    # q2_value = self.critic_2(states)
    # min_qvalue = paddle.sum(probs * paddle.minimum(q1_value, q2_value), axis=1, keepdim=True)  # 直接根据概率计算期望
    # actor_loss = paddle.mean(-self.log_alpha.exp() * entropy - min_qvalue)
    # self.actor_optimizer.clear_grad()
    # actor_loss.backward()
    # self.actor_optimizer.step()

    # # 更新alpha值
    # alpha_loss = paddle.mean((entropy - target_entropy).detach() * self.log_alpha.exp())
    # self.log_alpha_optimizer.clear_grad()
    # alpha_loss.backward()
    # self.log_alpha_optimizer.step()

    # self.soft_update(self.critic_1, self.target_critic_1)
    # self.soft_update(self.critic_2, self.target_critic_2)
