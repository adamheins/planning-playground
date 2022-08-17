from src.sampling_based_algos.RRG import RRG
import time
from pgraph import UGraph
from src.env.GraphPlanner import GraphPlanner

class RRT(RRG):
    def __init__(self, workspace, q0):
        super().__init__(workspace, q0)
        self.v_start.parent = None
        self.v_goal = None

    def query(
        self,
        start,
        goal,
        n=1000,
        min_edge_len=0.5,
        max_edge_len=1,
    ):
        new_size = self.graph.n + n
        start_time = time.time()
        while self.graph.n < new_size:
            # TODO maybe change min and max edge len based on how many points have been taken?
            # add radius with minimum path
            self.expand_graph(min_edge_len, max_edge_len)
            if self.end_condition(goal):
                self.preprocessing_time = 0
                self.query_time = time.time() - start_time
                break
        self.preprocessing_time = 0
        self.query_time = time.time() - start_time

    def expand_graph(self, min_edge_len, max_edge_len):
        """Add a vertex to the RRT graph"""
        q = self.workspace.sample()

        v_nearest, dist = self.closest_vertex(q)
        # if dist < min_edge_len:
        #     return
        q_nearest = v_nearest.coord

        # move toward q as much as possible
        if dist > max_edge_len:
            q = q_nearest + (q - q_nearest) * max_edge_len / dist
        if self.workspace.point_is_in_collision(q):
            q = self.closest_point_not_in_collision(q, q_nearest)
        # don't add if edge is in collision
        if self.workspace.edge_is_in_collision(q_nearest, q):
            return

        v = self.graph.add_vertex(q)
        v.connect(v_nearest)
        v.parent = v_nearest

    def end_condition(self, goal, goal_dist=1):
        v_nearest, dist = self.closest_vertex(goal)
        if dist <= goal_dist and not self.workspace.edge_is_in_collision(
            v_nearest.coord, goal
        ):
            v = self.graph.add_vertex(goal)
            v.connect(v_nearest)
            v.parent = v_nearest
            self.v_goal = v
            return True
        return False   
    
    def find_path(self):
            t = GraphPlanner(UGraph())
            t.graph.add_vertex(self.v_goal)
            if self.v_goal ==None:
                return None
            current = self.v_goal
            while current.parent != None:
                new_node = current.parent
                t.graph.add_vertex(new_node)
                t.graph.add_edge(current,new_node)
                current = current.parent
            return t