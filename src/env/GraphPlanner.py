import time
from pgraph import UGraph, UVertex
import numpy as np

class GraphPlanner:
    """Base class for all graph-based planners."""

    def __init__(self, graph):
        # graph may be directed or undirected, as selected by the subclass
        self.graph = graph

    def draw(self, ax, vertices=True, edges=True, rgb=(1, 0.5, 0)):
        """Draw the graph."""
        if vertices:
            for vertex in self.graph:
                x, y = vertex.coord
                ax.plot(x, y, "x", color=rgb)

        if edges:
            for edge in self.graph.edges():
                v1, v2 = edge.endpoints
                ax.plot(
                    [v1.coord[0], v2.coord[0]],
                    [v1.coord[1], v2.coord[1]],
                    color=(rgb[0], rgb[1], rgb[2], 0.25),
                )

    def k_nearest_neighbours(self, v, k, vertices=None):
        """Get k nearest neighbours to vertex v."""
        # get distance to each vertex in the graph
        if vertices is None:
            vertices = self.graph

        dists = []
        for vo in vertices:
            if v is vo:
                continue
            dists.append((vo.distance(v.coord), vo))

        # sort nearest to farthest
        dists.sort(key=lambda x: x[0])

        # return the k closest ones
        return [vo for _, vo in dists[:k]]

    def neighbours_within_dist(self, v, dist, vertices=None):
        """Get all vertices within dist of vertex v."""
        if vertices is None:
            vertices = self.graph

        dists = []
        for vo in vertices:
            if tuple(v) == tuple(vo.coord):
                continue
            dists.append((vo.distance(v), vo))

        dists.sort(key=lambda x: x[0])

        # return all neighbours with distance dist
        return [vo for d, vo in dists if d <= dist]

    def closest_vertex(self, q, vertices=None):
        """Get closest vertex in the graph, or a subset of vertices."""
        # this generalized the pgraph implementation, which doesn't handle a
        # subset of vertices
        if vertices is None:
            vertices = self.graph

        min_dist = vertices[0].distance(q)
        v_closest = vertices[0]

        for v in vertices:
            d = v.distance(q)
            if d < min_dist:
                min_dist = d
                v_closest = v

        return v_closest, min_dist

    def closest_vertex_to_graph(self, T):
        """Find the closest vertex to a another given graph T."""
        vertices_graph_T = T.graph
        vertices_graph = self.graph
        min_dist = vertices_graph[0].distance(vertices_graph_T[0].coord)
        v_closest = vertices_graph[0]
        v_closest_T = vertices_graph_T[0]
        for vertex in vertices_graph_T:
            for v in vertices_graph:
                d = vertex.distance(v.coord)

                if d < min_dist:
                    min_dist = d
                    v_closest = v
                    v_closest_T = vertex
        return v_closest, v_closest_T, min_dist

    def query(self, start, goal):
        """Get the shortest path between the start and goal points, if one exists."""
        start_time = time.time()
        vs, _ = self.graph.closest(start)
        vg, _ = self.graph.closest(goal)
        path = self.graph.path_Astar(vs, vg)
        if path is None:
            return None

        self.query_time = time.time() - start_time
        idx = path[0]
        points = np.array([self.graph[i].coord for i in idx])
        points = np.vstack((start, points, goal))
        t = GraphPlanner(UGraph())
        points = [UVertex(name=str(i), coord=i) for i in points]
        t.graph.add_vertex(points[0])
        for i in range(1, len(points)):
            t.graph.add_vertex(points[i])
            t.graph.add_edge(points[i - 1], points[i])
        return t
