#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# Modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size, action_dim, state_dim
                 , use_ld=False, low_act_dim=25, high_act_dim=25
        ):
        self.buffer = collections.deque(maxlen=max_size)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.DQN_action = []
        self.use_ld = use_ld
        self.low_act_dim = low_act_dim
        self.high_act_dim = high_act_dim

    def append(self, exp):
        self.buffer.append(exp)
        # if len(self.buffer) < 20 or len(self.buffer) % 10 == 0:
        #     self.countRes(alls)

    def append_DQN_action(self, act):
        self.DQN_action.append(act)

    def pop(self):
        return self.buffer.popleft()

    # 从经验池中选取N条经验出来
    def DQN_sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        cnt = 0
        for experience in mini_batch:
            s, a, r, s_p, done = experience
            DQN_a = self.DQN_action[cnt]
            cnt += 1
            obs_batch.append(s)
            action_batch.append(DQN_a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
            np.array(next_obs_batch).astype('float32'), np.array(
                done_batch).astype('float32')

            
    def exp_fast_learn_sample(self, batch_size):
        
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        batch_last_action = []
        
        batch_trend = []

        for experience in mini_batch:
            s, a, r, s_p, done, l_a, trend = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            batch_last_action.append(l_a)
            batch_trend.append(trend)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32').reshape(batch_size, self.action_dim), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32'), \
            np.array(batch_last_action).astype('float32').reshape(batch_size, self.high_act_dim), \
            np.array(batch_trend).astype('float32').reshape(batch_size, self.high_act_dim)
            
    def exp_fast_learn_sample_for_pca(self, batch_size):
        print("[exp_fast_learn_sample_for_pca] rpm act dim = ", self.action_dim)
        # mini_batch = random.sample(self.buffer, batch_size)
        mini_batch_idx = random.sample([i for i in range(len(self.buffer))], batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        batch_last_action = []
        
        batch_trend = []

        for i in mini_batch_idx:
            s, a, r, s_p, done, l_a, trend = self.buffer[i]
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            print("[exp_fast_learn_sample_for_pca] last action dim = ", l_a.shape)
            batch_last_action.append(l_a)
            batch_trend.append(trend)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32').reshape(batch_size, self.action_dim), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32'), \
            np.array(batch_last_action).astype('float32').reshape(batch_size, self.high_act_dim), \
            np.array(batch_trend).astype('float32').reshape(batch_size, self.high_act_dim), \
            mini_batch_idx
    
    def exp_fast_learn_sample_for_fix(self, mini_batch_idx):
        # mini_batch = random.sample(self.buffer, batch_size)
        # mini_batch_idx = random.sample([i for i in range(len(self.buffer))], batch_size)
        batch_size = len(mini_batch_idx)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        batch_last_action = []
        
        batch_trend = []

        for i in mini_batch_idx:
            s, a, r, s_p, done, l_a, trend = self.buffer[i]
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            batch_last_action.append(l_a)
            batch_trend.append(trend)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32').reshape(batch_size, self.action_dim), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32'), \
            np.array(batch_last_action).astype('float32').reshape(batch_size, self.high_act_dim), \
            np.array(batch_trend).astype('float32').reshape(batch_size, self.action_dim), \
            mini_batch_idx
            
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        batch_last_action = []

        for experience in mini_batch:
            s, a, r, s_p, done, l_a = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            batch_last_action.append(l_a)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32').reshape(batch_size, self.action_dim), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32'), \
            np.array(batch_last_action).astype(
                'float32').reshape(batch_size, self.high_act_dim)
            
    def sample_for_transfer(self, idx_list):
        
        
        batch_size = len(idx_list)
            
        
        # mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        batch_last_action = []

        for i in idx_list:
            s, a, r, s_p, done, l_a = self.buffer[i]
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            batch_last_action.append(l_a)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32').reshape(batch_size, self.action_dim), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32'), \
            np.array(batch_last_action).astype(
                'float32').reshape(batch_size, self.high_act_dim)
    
    def sample_o(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        batch_last_action = []

        for experience in mini_batch:
            s, a, r, s_p, done, l_a = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            batch_last_action.append(l_a)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32').reshape(batch_size, self.action_dim), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32'), \
            np.array(batch_last_action).astype(
                'float32').reshape(batch_size, self.action_dim)
                        
    def sample_for_pca(self, batch_size):
        mini_batch_idx = random.sample([i for i in range(len(self.buffer))], batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        batch_last_action = []

        for i in mini_batch_idx:
            s, a, r, s_p, done, l_a = self.buffer[i]
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            batch_last_action.append(l_a)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32').reshape(batch_size, self.action_dim), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32'), \
            np.array(batch_last_action).astype(
                'float32').reshape(batch_size, self.action_dim), \
            mini_batch_idx
                        
    def sample_for_fix(self, mini_batch_idx):
        batch_size = len(mini_batch_idx)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        batch_last_action = []

        for i in mini_batch_idx:
            s, a, r, s_p, done, l_a = self.buffer[i]
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
            batch_last_action.append(l_a)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32').reshape(batch_size, self.action_dim), \
            np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32'), \
            np.array(batch_last_action).astype(
                'float32').reshape(batch_size, self.action_dim)
            
            

    def __len__(self):
        return len(self.buffer)

    def getStates(self):
        obs_all = []
        for experience in self.buffer:
            if len(experience) == 6:
                s, a, r, s_p, done, l_a = experience
            elif len(experience) == 7:
                s, a, r, s_p, done, l_a, t = experience
            obs_all.append(s)
        # 获得state矩阵：len 行， state_dim 列
        # print('rpm state_dim = ', self.state_dim)
        obs_all = np.array(obs_all).astype('float32')
        # print('obs_all = ', obs_all)
        return obs_all

    def getActions(self):
        act_all = []
        for experience in self.buffer:
            s, a, r, s_p, done, l_a = experience
            act_all.append(a)
        # 获得state矩阵：len 行， state_dim 列
        act_all = np.array(act_all).astype('float32')
        return act_all

    def countRes(self, alls):
        # 分别计算alls矩阵的mean和std
        mean = np.mean(alls, axis=0)
        std = np.std(alls, axis=0)
        # print('mean:', mean)
        # print('std:', std)

        # avoid std[i] == 0
        for i in range(len(std)):
            # self.mean[i] = np.mean(obs_all[:, i])
            # self.std[i] = np.std(obs_all[:, i])
            if std[i] == 0:
                std[i] = 1

        return mean, std
