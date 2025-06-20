from collections import defaultdict

class NumMapper:
    def __init__(self):
        """初始化一个空的映射数据结构"""
        self.data = defaultdict(set)  # 使用 set 存储 num2，避免重复

    def add(self, num1, num2):
        """
        添加一个 (num1, num2) 映射
        """
        self.data[num1].add(num2)

    def remove(self, num1, num2):
        """
        删除一个 (num1, num2) 映射
        """
        if num1 in self.data:
            self.data[num1].discard(num2)  # 删除 num2
            if not self.data[num1]:  # 如果 num1 对应的集合为空，删除 num1
                del self.data[num1]

    def get_all(self, num1):
        """
        获取 num1 对应的所有 num2
        """
        return self.data[num1] if num1 in self.data else set()

    def add_batch(self, num1, nums2):
        """
        批量添加多个 num2 到 num1 的映射
        """
        self.data[num1].update(nums2)

    def remove_batch(self, num1, nums2):
        """
        批量删除多个 num2 从 num1 的映射
        """
        if num1 in self.data:
            self.data[num1].difference_update(nums2)
            if not self.data[num1]:  # 如果 num1 对应的集合为空，删除 num1
                del self.data[num1]

    def exists(self, num1, num2):
        """
        判断 (num1, num2) 是否存在
        """
        return num1 in self.data and num2 in self.data[num1]

    def __repr__(self):
        """
        打印当前数据的表示形式
        """
        return str(dict(self.data))  # 将 defaultdict 转为普通字典打印
