import numpy as np
from hes.num_mapper import NumMapper

class LowDimAdaptor:
    def __init__(self, env, target_dim, seed=None, high_dim=25):
        """
        初始化 lowDimAdapter 类
        :param env: 包含高维参数信息的环境数据（例如 env 实例）
        :param target_dim: 低维向量 d 的维度
        :param seed: 随机种子，保证生成的投影矩阵可复现
        """
        if env is not None:
            self.action_info = env.action_info  # 高维参数信息
            self.high_dim = env.action_dim  # 高维参数的维度，假设为 25
        else:
            self.high_dim = high_dim
        self.target_dim = target_dim  # 低维向量 d 的维度
        self.seed = seed
        
        # 随机数生成器
        self.rs = np.random.RandomState(seed)
        
        # 初始化低维向量 d 和投影矩阵 A
        self.d = np.zeros(target_dim)
        self.A = self._generate_projection_matrix()
        self.A_pseudo_inv = np.linalg.pinv(self.A)

    def _generate_projection_matrix(self):
        """
        生成随机投影矩阵 A，其形状为 (高维度, 低维度)
        确保每列只有一个值为 +1 或 -1，其余为 0
        :return: 随机投影矩阵 A
        """
        A = np.zeros((self.high_dim, self.target_dim))

        for i in range(self.high_dim):
            # 随机选择一个行索引 i，将 A[i, j] 设置为 +1 或 -1
            j = self.rs.choice(self.target_dim)
            A[i, j] = self.rs.choice([-1, 1])

        return A

    def transform(self, model_d):
        """
        将低维向量 d 转化为高维参数 D
        :return: 高维参数向量 D
        """
        # 使用矩阵乘法将 d 转换为 D
        # print("self.d = ", self.d)
        # print("model.d = ", model_d)
        D = np.dot(self.A, model_d)
        return D
    
    def batch_transform(self, model_d):
        """
        将低维向量 d 转化为高维参数 D
        :return: 高维参数向量 D
        """
        # 检查输入是否为批量数据
        if model_d.ndim == 1:  # 如果是单样本数据
            model_d = np.expand_dims(model_d, axis=0)  # 增加批量维度，变为 [1, input_dim]

        # 使用批量矩阵乘法
        D = np.matmul(model_d, self.A.T)  # self.A 的形状为 [output_dim, input_dim]
        return D
    
    def reverse_transform(self, model_D):
        d = np.dot(self.A_pseudo_inv, model_D)
        return d

    def set_d(self, new_d):
        """
        更新低维向量 d 的值
        :param new_d: 新的低维向量值
        """
        if len(new_d) != self.target_dim:
            raise ValueError(f"new_d 的维度应为 {self.target_dim}")
        self.d = new_d

    def get_d(self):
        """
        获取当前的低维向量 d
        :return: 当前的低维向量 d
        """
        return self.d

    def get_D(self):
        """
        获取当前低维向量 d 转换得到的高维向量 D
        :return: 当前的高维向量 D
        """
        return self.transform()
    
    def test(self, model_D):
        d = np.dot(self.A_pseudo_inv, model_D)
        print("d = A-1 * D = ", d)
        D_new = np.dot(self.A, d)
        print("D = A * d = ", D_new)
        diff = model_D - D_new
        print("diff = D_o - D = ", diff)
        
    def double_network_correct(self, A_1):
        target_num_act = [i+1 for i in range(self.target_dim)]
        target_num_neg = [-(i+1) for i in range(self.target_dim)]
        print("[DOUBLE NETWORK CORRECT] target_num_act = ", target_num_act)
        print("[DOUBLE NETWORK CORRECT] target_num_neg = ", target_num_neg)
        
        
        mapper = NumMapper()
        for x in target_num_act:
            for y in target_num_act:
                # if x > y:   continue
                mapper.add(x, y)
                mapper.add(x, -y)
        print("[DOUBLE NETWORK CORRECT] init map:", mapper)
        
        
        print("[DOUBLE NETWORK CORRECT] anothor A = ", A_1)
        if self.target_dim != self.high_dim or A_1.shape[0] != self.target_dim:
            for i in range(A_1.shape[0]):
                
                nonzero_indices_1 = [index for index, value in enumerate(A_1[i]) if value != 0][0]+1
                nonzero_indices_2 = [index for index, value in enumerate(self.A[i]) if value != 0][0]+1
                non_zero_sign = 1
                
                if A_1[i][nonzero_indices_1-1] * self.A[i][nonzero_indices_2-1] < 0:
                    non_zero_sign = -1
                min_indices = abs(nonzero_indices_1) #min(abs(nonzero_indices_1), abs(nonzero_indices_2))
                max_indices = abs(nonzero_indices_2) #max(abs(nonzero_indices_1), abs(nonzero_indices_2))
                if mapper.exists(min_indices, max_indices * non_zero_sign):
                    mapper.remove(min_indices, max_indices * non_zero_sign)
                    mapper.remove(max_indices, min_indices * non_zero_sign)
                else:
                    print("[DOUBLE NETWORK CORRECT] find conflict: ({0}, {1})".format(min_indices, max_indices*non_zero_sign))
                    print("[DOUBLE NETWORK CORRECT] find remain candidates of {0}: {1}".format(min_indices, mapper.get_all(min_indices)))
                    candidate = list(mapper.get_all(min_indices))
                    if len(candidate) > 0:
                        print("[DOUBLE NETWORK CORRECT] old A[i] = ", self.A[i])
                        rnd_idx = self.rs.choice(len(candidate))
                        self.A[i][max_indices-1] = 0
                        new_sign = 1
                        
                        if candidate[rnd_idx] * A_1[i][nonzero_indices_1-1] < 0:
                            new_sign = -1
                        print("[DOUBLE NETWORK CORRECT] A_1[i][nonzero_indices_1-1] = ", A_1[i][nonzero_indices_1-1])
                        
                        print("[DOUBLE NETWORK CORRECT] select new index = {0}, new sign = {1}".format(candidate[rnd_idx], new_sign))

                        self.A[i][abs(candidate[rnd_idx])-1] = new_sign
                        print("[DOUBLE NETWORK CORRECT] new A[i] = ", self.A[i])
                        mapper.remove(min_indices, abs(candidate[rnd_idx]) * new_sign)
                        mapper.remove(abs(candidate[rnd_idx]), min_indices * new_sign)
                
                
                
                print("[DOUBLE NETWORK CORRECT] final pair: ({0}, {1})".format(min_indices, max_indices*non_zero_sign))
        elif self.target_dim == self.high_dim:
            self.A = np.eye(self.high_dim)      
                # if mapper.get_all()
            
        
        
        
        
        # target_num_mix = target_num + ()











