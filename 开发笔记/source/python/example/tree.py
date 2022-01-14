class Solution:
    def find_solution(self, n: int):
        """
        输入节点数，输出树的类别数
        :param n: node number
        :return:
        """
        # 已有解决方案则直接取值
        if self.solution[n] != 0:
            return self.solution[n]
        else:
            sub_tree_node_num = n - 1
            solution_num = 0
            for i, j in self.addition_factor(sub_tree_node_num):
                solution_num = solution_num + self.find_solution(i) * self.find_solution(j)
            self.solution[n] = solution_num
            return solution_num

    def numTrees(self, n: int) -> int:
        # memorize the solution
        self.solution = [0 for _ in range(n + 1)]
        # 节点数为0或1时，树类为1
        self.solution[0] = 1
        self.solution[1] = 1

        self.find_solution(n)
        return self.solution[n]

    def addition_factor(self, n):
        """
        给定一个正整数，给出其加法对
        \n e.g. n=2, 输出为(0,2)(1,1)(2,0)
        :param int n:
        :return:
        """
        for i in range(n + 1):
            yield i, n - i


if __name__ == '__main__':
    solution = Solution()
    print(solution.numTrees(4))
