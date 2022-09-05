import numpy as np
from pgraph import UGraph
from .rrt import RRT
import time


class RRT_star(RRT):
    def __init__(self, workspace, q0):
        super().__init__(workspace, q0)

    def vertex_cost(self, v):
        """Determine the cost to reach vertex v from the start."""
        cost = 0

        while v.parent is not None:
            cost += self.edge_cost(v.coord, v.parent.coord)
            v = v.parent

        return cost

    def edge_cost(self, qa, qb):
        """Return the cost of connecting two vertices"""
        return np.linalg.norm(qa - qb)

    def add_vertex(self, v_nearest, q, neighborhood_radius, neighborhood=None, rewire=True):
        """Add a vertex located at q to the graph connected to existing vertex parent."""
        t = time.time()
        v = self.graph.add_vertex(q)

        # nominal cost of connecting to nearest neighbor
        cost = self.vertex_cost(v_nearest) + self.edge_cost(v_nearest.coord, q)
        cap = 5
        if neighborhood is None:
            neighborhood = self.neighbours_within_dist(q, neighborhood_radius, cap)
        else:
            neighborhood = neighborhood[:min(cap,len(neighborhood))]

        # see if cost is reduced by connecting to any of the vertices in the neighborhood
        for v_near in neighborhood:
            if not self.workspace.edge_is_in_collision(q, v_near.coord):
                new_cost = self.vertex_cost(v_near) + self.edge_cost(v_near.coord, q)
                if new_cost < cost:
                    cost = new_cost
                    v_nearest = v_near

        # connect
        v.parent = v_nearest
        v.edge = self.graph.add_edge(v_nearest, v)

        if rewire:
            self.rewire(v, neighborhood)
        #print("adding time",time.time()-t)
        return v

    def extend(
        self,
        goal,
        n=1000,
        min_edge_len=0.5,
        max_edge_len=1,
        niu=1,
        divide_edges=True,
        stop_early=True,
    ):
        """Extend the tree with up to n nodes.

        If `stop_early` is passed, return as soon as a path to the goal is found.
        If `divide_edges` is passed, then instead of discarding samples farther
        than `max_edge_len` from the graph, multiples vertices are added such
        that the edges between each are less than `max_edge_len`.

        Returns True if a path to the goal exists, False otherwise.
        """
        count = 0
        prev= 0
        sample = 1
        #t = time.time()
        while count < n:
            # if count!= prev:
            #     #print(count)
            #     print(prev,sample)
            #     prev= count
            #     sample = 1
                #t = time.time()
            q = self.workspace.sample()
            
            neighborhood = self.neighbours_within_dist(q, max(self.workspace.width,self.workspace.height), cap=10)
            v_nearest = neighborhood[0]
            dist = self.edge_cost(q,v_nearest.coord)

            # if new vertex would be too far away, split it so that each new
            # vertex is within max_edge_len of each other
            if divide_edges and dist > max_edge_len:
                num_samples = round(dist / max_edge_len)
                qs = []
                for i in range(num_samples):
                    step = (num_samples - i) / num_samples
                    qs.append(v_nearest.coord + step * (q - v_nearest.coord))
            else:
                qs = [q]

            # process all samples
            while len(qs) > 0:
                #t = time.time()
                # print(count)
                q = qs.pop()
                v_nearest, dist = self.closest_vertex(q, vertices=neighborhood)
                q = self.steer(q, v_nearest, dist, niu, min_edge_len)
                if q is None:
                    continue
                count += 1
                # add the new vertex
                radius = min(1 * (np.log(count) / count) ** (0.5), min_edge_len)
                v = self.add_vertex(v_nearest, q, radius, neighborhood=neighborhood, rewire=True)
                neighborhood.append(v)

                # try to connect to the goal if we don't already have a path
                if not self.has_path_to_goal():
                    if v.distance(
                        goal
                    ) <= max_edge_len and not self.workspace.edge_is_in_collision(
                        q, goal
                    ):
                        self.v_goal = self.add_vertex(
                            v, goal, max_edge_len, rewire=True
                        )
                        count += 1

                # if we just want a path and don't want to iterate further, stop
                # now
                if self.has_path_to_goal() and stop_early:
                    return True
                #print(time.time() - t)
            sample +=1
        return self.has_path_to_goal()

    def rewire(self, v, neighborhood):
        """Rewire edges in the neighborhood to minimize cost.

        In particular, check if the cost to reach each node would be lower if
        connected through the new node v."""
        cost_v = self.vertex_cost(v)
        for v_near in neighborhood:
            # check if we can connect at all
            if self.workspace.edge_is_in_collision(v.coord, v_near.coord):
                continue

            cost = cost_v + self.edge_cost(v.coord, v_near.coord)
            if self.vertex_cost(v_near) > cost:
                self.graph.remove(v_near.edge)
                v_near.parent = v
                v_near.edge = self.graph.add_edge(v, v_near)
