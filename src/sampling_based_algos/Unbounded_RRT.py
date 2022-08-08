from src.sampling_based_algos.RRT import RRT
import time
import numpy as np

class Unbounded_RRT(RRT):
    """RRT with no minimum distance"""

    def __init__(self, workspace, q0):
        super().__init__(workspace, q0)

    def query(self, start, goal, n=1000, min_edge_len=0.5, max_edge_len=1, niu=0.8):
        new_size = self.graph.n + n
        start_time = time.time()
        while self.graph.n < new_size:
            # TODO maybe change min and max edge len based on how many points have been taken?
            # add radius with minimum path
            self.expand_graph(min_edge_len, max_edge_len, niu)
            if self.end_condition(goal):
                self.preprocessing_time = 0
                self.query_time = time.time() - start_time
                break
        self.preprocessing_time = 0
        self.query_time = time.time() - start_time

    def expand_graph(self, min_edge_len, max_edge_len, niu=0.8):
        """Add a vertex to the RRT graph"""
        q = self.workspace.sample()
        v_nearest, dist = self.closest_vertex(q)
        q_nearest = v_nearest.coord

        # steer the point closer to the tree
        q = q + (q_nearest - q) * niu / dist
        dist = dist - niu
        if dist < min_edge_len:
            return

        if self.workspace.point_is_in_collision(q):
            q = self.closest_point_not_in_collision(q, q_nearest)
        # don't add if edge is in collision
        if self.workspace.edge_is_in_collision(q_nearest, q):
            return

        # split distance into multiple edges
        if dist > max_edge_len:
            q_closest = v_nearest.coord
            vertices = round(dist / max_edge_len)
            for i in range(vertices - 1):
                cur_vertex = q_closest + (i + 1) * (q - q_closest) / np.linalg.norm(
                    q - q_closest
                )
                cur_vertex = self.graph.add_vertex(cur_vertex)
                cur_vertex.connect(v_nearest)
                cur_vertex.parent = v_nearest
                v_nearest = cur_vertex

        v = self.graph.add_vertex(q)
        v.connect(v_nearest)
        v.parent = v_nearest