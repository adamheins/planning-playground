from src.sampling_based_algos.RRG import RRG
import time


class Bidirectional_RRT(RRG):
    def __init__(self, workspace, q0, rrt_cls):
        super().__init__(workspace, q0)
        self.rrt_cls = rrt_cls
        self.ta = None
        self.tb = None

    def query(self, start, goal, k=1000, min_edge_len=0.5, max_edge_len=1):
        """Use two trees, one starting from start one from goal, to explore the space"""
        start_time = time.time()
        ta = self.rrt_cls(self.workspace, start)
        tb = self.rrt_cls(self.workspace, goal)
        v_ta = ta.graph.add_vertex(start)
        v_tb = tb.graph.add_vertex(goal)
        for _ in range(k):
            ta.expand_graph(min_edge_len, max_edge_len)

            # operate on second tree
            tb.expand_graph(min_edge_len, max_edge_len)

            if self.touch(ta, tb):
                self.preprocessing_time = 0
                self.query_time = time.time() - start_time
                self.ta = ta
                self.tb = tb
                return
            if ta.graph.n - tb.graph.n > 10:
                ta, tb = tb, ta
        self.preprocessing_time = 0
        self.query_time = time.time() - start_time
        self.ta = None
        self.tb = None

    def touch(self, ta, tb, goal_dist=0.5):
        """determine if two trees are touching each other"""
        v, v_tb, dist = ta.closest_vertex_graph(tb)
        if dist < goal_dist:
            v.connect(v_tb)
            ta.graph.add_edge(v, v_tb)
            ta.v_goal = v
            tb.v_goal = v_tb
            return True
        return False

    def draw(self, ax):
        if self.ta is None:
            print("No solution")
        else:
            self.ta.draw(ax, rgb=(0, 1, 1))
            self.tb.draw(ax, rgb=(1, 1, 0))

    def find_path(self):
        path_ta = self.ta.find_path()
        path_tb = self.tb.find_path()
        v_ta ,v_tb, _ = path_ta.closest_vertex_graph(path_tb)
        current = v_tb
        previous = v_ta
        while previous.parent != None:
            new = path_ta.graph.add_vertex(current.coord)
            new.connect(previous)
            previous = current
            current = current.parent
        return path_ta