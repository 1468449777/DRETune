import parl
from parl import layers
import numpy as np
import paddle
from paddle import fluid


LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


class SACAlphaModel(parl.Model):
    def __init__(self):
        # 定义模型结构
        hid1_size = 256
        hid2_size = 256

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.alpha_output = layers.fc(size=1, act=None)

    def forward(self, obs):
        # 前向传播定义
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        alpha = self.alpha_output(hid2)
        alpha = layers.squeeze(alpha, axes=[1])
        alpha = layers.exp(alpha)  # 确保alpha为正值
        # print("[ALPHA LOG] use alpha model func [forward], alpha = {0}".format(alpha))

        return alpha


class SACRewardModel(parl.Model):
    def __init__(self):
        # 定义模型结构
        hid1_size = 256
        hid2_size = 256

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.mean_linear = layers.fc(size=hid2_size)
        self.log_std_linear = layers.fc(size=hid2_size)

    def forward(self, obs_act):
        # 前向传播定义
        hid1 = self.fc1(obs_act)
        hid2 = self.fc2(hid1)
        means = self.mean_linear(hid2)
        log_std = self.log_std_linear(hid2)
        log_std = layers.clip(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return means, log_std

# code intrinsic_reward


class SACStateModel(parl.Model):
    def __init__(self, state_dim):
        # 定义模型结构
        hid1_size = 256
        hid2_size = 512
        hid3_size = 1024
        hid4_size = 512
        hid5_size = 256

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=hid3_size, act='relu')
        self.fc4 = layers.fc(size=hid4_size, act='relu')
        self.fc5 = layers.fc(size=hid5_size, act='relu')
        self.output = layers.fc(size=state_dim)

    def forward(self, act):
        # 前向传播定义
        hid1 = self.fc1(act)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        hid4 = self.fc4(hid3)
        hid5 = self.fc5(hid4)
        output = self.output(hid5)

        return output


class SACActorModel(parl.Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim
        hid1_size = 128
        hid2_size = 256
        hid3_size = 128
        trend_size = act_dim
        hid4_size = 128
        hid5_size = 256
        hid6_size = 128
        
        # 趋势预测网络
        self.fc1 = layers.fc(size=hid1_size, act='relu', param_attr=fluid.initializer.Xavier(uniform=True))
        self.fc2 = layers.fc(size=hid2_size, act='relu', param_attr=fluid.initializer.Xavier(uniform=True))
        self.fc3 = layers.fc(size=hid3_size, act='relu', param_attr=fluid.initializer.Xavier(uniform=True))
        # self.tr  = layers.fc(size=trend_size, act='sigmoid', param_attr=fluid.initializer.Xavier(uniform=True))
        # self.tr  = layers.fc(size=trend_size, act='tanh', param_attr=fluid.initializer.Xavier(uniform=True))
        # self.tr  = layers.fc(size=trend_size, act='relu', param_attr=fluid.initializer.Xavier(uniform=True))
        self.trend_logits = layers.fc(size=3 * act_dim, param_attr=fluid.initializer.Xavier(uniform=True))  # 三分类 logits: 对应 -1, 0, 1
        
        # 幅度预测网络
        self.fc4 = layers.fc(size=hid4_size, act='relu')
        self.fc5 = layers.fc(size=hid5_size, act='relu')
        self.fc6 = layers.fc(size=hid6_size, act='relu')
        self.amplitude = layers.fc(size=act_dim, act='sigmoid')  # 幅度输出范围 [0, 1]
        self.log_std_linear = layers.fc(size=act_dim)
            
            
    def trend_policy(self, obs, temperature=0.5):
        # 趋势预测网络前向传播
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        # hidtr = self.tr(hid3)
        logits = self.trend_logits(hid3)  # 输出 logits
        # 输出层，预测分类概率
        # logits = fluid.layers.fc(input=hidden, size=action_dim * 3, act=None)  # 每个动作维度 3 个类别
        logits = fluid.layers.reshape(logits, shape=[-1, self.act_dim, 3])  # 调整为 [batch_size * action_dim, 3]
        trend = fluid.layers.argmax(logits, axis=-1)
        return logits, trend


    def amplitude_policy(self, obs):
        # 幅度预测网络前向传播
        hid4 = self.fc4(obs)
        hid5 = self.fc5(hid4)
        hid6 = self.fc6(hid5)
        amplitude = self.amplitude(hid6)  # 输出 [0, 1]
        log_std = self.log_std_linear(hid6)
        log_std = layers.clip(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return amplitude, log_std


    def policy(self, obs):
        # 趋势预测
        trend_values, trend = self.trend_policy(obs)
        trend = layers.cast(trend-1.0, dtype='float32')
        
        # 幅度预测
        amplitude, log_std = self.amplitude_policy(obs)
        
        # 组合输出
        combined_output = layers.elementwise_mul(trend, amplitude)  # [-1, 1]
        return combined_output, trend_values, amplitude, log_std



class SACActorModelDouble(parl.Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim
        hid1_size = 128
        hid2_size = 256
        hid3_size = 128
        trend_size = act_dim
        hid4_size = 128
        hid5_size = 256
        hid6_size = 128
        
        # 趋势预测网络
        self.fc1 = layers.fc(size=hid1_size, act='relu', param_attr=fluid.initializer.Xavier(uniform=True))
        self.fc2 = layers.fc(size=hid2_size, act='relu', param_attr=fluid.initializer.Xavier(uniform=True))
        self.fc3 = layers.fc(size=hid3_size, act='relu', param_attr=fluid.initializer.Xavier(uniform=True))
        # self.tr  = layers.fc(size=trend_size, act='sigmoid', param_attr=fluid.initializer.Xavier(uniform=True))
        # self.tr  = layers.fc(size=trend_size, act='tanh', param_attr=fluid.initializer.Xavier(uniform=True))
        # self.tr  = layers.fc(size=trend_size, act='relu', param_attr=fluid.initializer.Xavier(uniform=True))
        # self.trend_logits = layers.fc(size=3 * act_dim, param_attr=fluid.initializer.Xavier(uniform=True))  # 三分类 logits: 对应 -1, 0, 1
        self.trend_double = layers.fc(size=act_dim, act='sigmoid', param_attr=fluid.initializer.Xavier(uniform=True))
        
        # 幅度预测网络
        self.fc4 = layers.fc(size=hid4_size, act='relu')
        self.fc5 = layers.fc(size=hid5_size, act='relu')
        self.fc6 = layers.fc(size=hid6_size, act='relu')
        self.amplitude = layers.fc(size=act_dim, act='sigmoid')  # 幅度输出范围 [0, 1]
        self.log_std_linear = layers.fc(size=act_dim)
            
            
    def trend_policy(self, obs, temperature=0.5):
        # 趋势预测网络前向传播
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        # hidtr = self.tr(hid3)
        # logits = self.trend_logits(hid3)  # 输出 logits
        # # 输出层，预测分类概率
        # # logits = fluid.layers.fc(input=hidden, size=action_dim * 3, act=None)  # 每个动作维度 3 个类别
        # logits = fluid.layers.reshape(logits, shape=[-1, self.act_dim, 3])  # 调整为 [batch_size * action_dim, 3]
        # trend = fluid.layers.argmax(logits, axis=-1)
        trend = self.trend_double(hid3)
        trend = (trend * 2 - 1) * 3
        return trend, trend


    def amplitude_policy(self, obs):
        # 幅度预测网络前向传播
        hid4 = self.fc4(obs)
        hid5 = self.fc5(hid4)
        hid6 = self.fc6(hid5)
        amplitude = self.amplitude(hid6)  # 输出 [0, 1]
        log_std = self.log_std_linear(hid6)
        log_std = layers.clip(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return amplitude, log_std


    def policy(self, obs):
        # 趋势预测
        trend_values, trend = self.trend_policy(obs)
        trend = layers.cast(trend-1.0, dtype='float32')
        
        # 幅度预测
        amplitude, log_std = self.amplitude_policy(obs)
        
        # 组合输出
        combined_output = layers.elementwise_mul(trend, amplitude)  # [-1, 1]
        return combined_output, trend_values, amplitude, log_std




    # def policy(self, obs):
    #     hid1 = self.fc1(obs)
    #     hid2 = self.fc2(hid1)
    #     hid3 = self.fc3(hid2)
    #     hidtr = self.tr(hid3)
    #     hidtr = hidtr * 2 - 1
    #     # hidtr = layers.tanh(hidtr / 1e-3)

            
            
        
    #     hid4 = self.fc4(obs)
    #     hid5 = self.fc5(hid4)
    #     hid6 = self.fc6(hid5)
    #     means = self.mean_linear(hid6)
    #     means = layers.elementwise_mul(means, hidtr)
        
        
    #     log_std = self.log_std_linear(hid6)
    #     log_std = layers.clip(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

    #     return means, log_std
    
    
    # def tr_policy(self, obs):
    #     hid1 = self.fc1(obs)
    #     hid2 = self.fc2(hid1)
    #     hid3 = self.fc3(hid2)
    #     hidtr = self.tr(hid3)
    #     # hidtr = hidtr * 2 - 1
    #     # hidtr = layers.round(hidtr)  # 离散化为 -1, 0, 1
    #     # hidtr = layers.tanh(hidtr / 1e-3)
    #     return hidtr









class SACCriticModel(parl.Model):
    
    def __init__(self):
        hid1_size = 400
        hid2_size = 300

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=1, act=None)

        self.fc4 = layers.fc(size=hid1_size, act='relu')
        self.fc5 = layers.fc(size=hid2_size, act='relu')
        self.fc6 = layers.fc(size=1, act=None)
# 这里也是输出2个Q值，Q1 Q2与TD3 相同

    def value(self, obs, act):
        hid1 = self.fc1(obs)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        hid2 = self.fc4(obs)
        concat2 = layers.concat([hid2, act], axis=1)
        Q2 = self.fc5(concat2)
        Q2 = self.fc6(Q2)
        Q2 = layers.squeeze(Q2, axes=[1])

        return Q1, Q2


class SACModel(parl.Model):
    def __init__(self, act_dim, state_dim, use_double=False):
        if use_double:
            self.actor_model = SACActorModelDouble(act_dim)
        else:  
            self.actor_model = SACActorModel(act_dim)
        self.critic_model = SACCriticModel()
        self.alpha_model = SACAlphaModel()
        self.reward_model = SACRewardModel()
        self.act_dim = act_dim

        self.states_model = SACStateModel(state_dim)

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    # def Q1(self, obs, act):
    #     return self.critic_model.Q1(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_alpha(self, obs):
        alpha = self.alpha_model.forward(obs)
        # print("[ALPHA LOG] use alpha model func [get_alpha], alpha = {0}".format(alpha))
        return alpha
