from src.sampling_based_algos.RRG import RRG
import numpy as np


class Bidirectional_RRT(RRG):
    def __init__(self, workspace, q0, rrt_cls):
        super().__init__(workspace, q0)
        self.rrt_cls = rrt_cls

    def extend(self, goal, n=100, min_edge_len=0.5, max_edge_len=1, niu=1,  stop_early=True, divide_edges=False ):
        """Use two trees, one starting from start one from goal, to explore the workspace"""
        ta = self.rrt_cls(self.workspace, self.v_start.coord)
        tb = self.rrt_cls(self.workspace, goal)
        self.ta = ta
        self.tb = tb
        while ta.graph.n + tb.graph.n < n:
            ta.extend(tb.v_start.coord, n=1, min_edge_len=min_edge_len, max_edge_len=max_edge_len, niu=niu, stop_early=stop_early, divide_edges=divide_edges)
            # operate on second tree
            tb.extend(ta.v_start.coord, n=1, min_edge_len=min_edge_len, max_edge_len=max_edge_len, niu=niu, stop_early=stop_early, divide_edges=divide_edges)

            self.ta = ta
            self.tb = tb
            if not self.has_path_to_goal():
                v, v_tb, dist = ta.closest_vertex_to_graph(tb)
                if dist < max_edge_len and not self.workspace.edge_is_in_collision(v.coord, v_tb.coord):
                    v.connect(v_tb)
                    ta.graph.add_edge(v, v_tb)
                    ta.v_goal = v
                    tb.v_goal = v_tb
            if ta.graph.n - tb.graph.n > 10:
                ta, tb = tb, ta

            if self.has_path_to_goal() and stop_early:
                return True

        return self.has_path_to_goal()
                

    def has_path_to_goal(self):
        """Return True if a path to the goal exists, False otherwise."""
        return self.ta.v_goal is not None
  

    def draw(self, ax):
        """Draw both graphs"""
        self.ta.draw(ax, rgb=(0, 1, 1))
        self.tb.draw(ax, rgb=(1, 1, 0))

    def get_path_to_goal(self):
        """Get the path to the goal node.

        Returns a 2D array of points representing the path or None if no path
        has been found."""
        if not self.has_path_to_goal():
            return None
        path_ta = self.ta.get_path_to_goal()[::-1]
        path_tb = self.tb.get_path_to_goal()
        return np.append(path_ta, path_tb,axis = 0)