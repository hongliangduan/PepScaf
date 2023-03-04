import time
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, track
import math
from conf import Mcts

c_param = Mcts.c_param

class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds

        Returns
        -------

        """

        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else:
            with Progress(SpinnerColumn(), *Progress.get_default_columns(), "Elapsed:",
                          TimeElapsedColumn()) as progress:
                for _ in progress.track(range(0, simulations_number), description='Searching...'):
                    v = self._tree_policy()
                    reward = v.rollout()
                    v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=c_param)

    def best_actions(self, simulations_number=None, total_simulation_seconds=None, n_best=2):
        """

                Parameters
                ----------
                simulations_number : int
                    number of simulations performed to get the best action

                total_simulation_seconds : float
                    Amount of time the algorithm has to run. Specified in seconds
                best_n : int
                    Number of n best for output

                Returns
                -------

                """
        if simulations_number is None:
            assert (total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else:
            with Progress(SpinnerColumn(), *Progress.get_default_columns(), "Elapsed:",
                          TimeElapsedColumn()) as progress:
                for _ in progress.track(range(0, simulations_number), description='Searching...'):
                    v = self._tree_policy()
                    reward = v.rollout()
                    v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_childs(c_param=c_param, n_best=n_best)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():  # 如果不是终点
            if not current_node.is_fully_expanded():  # 如果没有拓展完
                return current_node.expand()  # 那就进行拓展
            else:
                current_node = current_node.best_child()  # 否则就返回最好的子节点
        return current_node
