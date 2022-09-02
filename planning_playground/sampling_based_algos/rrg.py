from .graph_planner import GraphPlanner
import time
from pgraph import UGraph
import numpy as np


class RRG(GraphPlanner):
    """Rapidly-exploring random graph.

    Equivalent to a rapidly-explored random tree (RRT) if all vertices are
    added with connect_multiple_vertices=False.
    """

    def __init__(self, workspace, q0):
        super().__init__(UGraph())
        self.workspace = workspace
        self.v_start = self.graph.add_vertex(q0)
        self.v_goal = None

        # TODO include this properly, bias toward it, etc.
        # self.goal = self.graph.add_vertex(qf)

    def add_vertices(
        self,
        n,
        min_edge_len=0.5,
        max_edge_len=1,
        near_dist=1,  # neigbors distance
        connect_multiple_vertices=True,
    ):
        """Add n vertices to the RRG."""
        new_size = self.graph.n + n
        start_time = time.time()
        while self.graph.n < new_size:
            # TODO we can actually try to connect as close as possible to a
            # vertex in collision
            q = self.workspace.sample()

            v_nearest, dist = self.closest_vertex(q)

            if dist < min_edge_len:
                continue
            q_nearest = v_nearest.coord
            # move toward q as much as possible
            if dist > max_edge_len:
                q = q_nearest + (q - q_nearest) * max_edge_len / dist

            if self.workspace.point_is_in_collision(q):
                q = self.closest_point_not_in_collision(q, q_nearest)

            # don't add if edge is in collision
            if self.workspace.edge_is_in_collision(q_nearest, q):
                continue

            v = self.graph.add_vertex(q)
            v.connect(v_nearest)

            # find and add additional nearby vertices
            if connect_multiple_vertices:
                vs_near = self.neighbours_within_dist(v, near_dist)
                for vo in vs_near:
                    # don't make duplicate edges
                    if vo.isneighbour(v):
                        continue

                    if v.distance(vo.coord) < min_edge_len:
                        continue

                    # avoid collisions
                    if self.workspace.edge_is_in_collision(vo.coord, v.coord):
                        continue
                    vo.connect(v)
        self.preprocessing_time = time.time() - start_time

    def closest_point_not_in_collision(self, q, q_nearest):
        """return closest point that is not in collision with workspace"""
        while self.workspace.point_is_in_collision(q):
            q = q + (q - q_nearest) / np.linalg.norm(q - q_nearest)
        return q
