

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from pgraph import UGraph, DGraph, UVertex, Edge
import time
from planning import RRT, Workspace, Rectangle, Circle
import math

class RRT_star(RRT):
    def __init__(self, workspace, q0):
        self.graph = UGraph()
        self.workspace = workspace
        v = self.graph.add_vertex(q0)
        v.parent = None
        self.v_goal = None
        

    def cost(self,vb):
        """Determine the cost in between two points"""
        cost = 0
        parent = vb.parent
        while vb.parent != None:
            cost += self.connection(vb.coord,parent.coord)
            vb = vb.parent
            parent = vb.parent
        return cost
        

    def near_vertices(self,x,rn):
        """Return all vertexes belonging to the graph in the ball"""
        X_near = []
        for vertex in self.graph:
            q = vertex.coord
            #print(np.linalg.norm(x-q),rn)
            if np.linalg.norm(x-q) <= rn:
                X_near.append(vertex)
        return X_near
    
    def connection(self, va, vb):
        """Return the cost of connecting two verices"""
        return np.linalg.norm(va -vb)

    
    def query(
        self,
        start,
        goal,
        n=1000,
        min_edge_len=0.5,
        max_edge_len=1,
        niu = 0.8
    ):
        new_size = self.graph.n 
        start_time = time.time()
        samples = []
        prev = 0
        while new_size<n:
            # if prev !=new_size:
            #     print(new_size)
            #     prev = new_size
            # TODO maybe change min and max edge len based on how many points have been taken?
            # add radius with minimum path
            if len(samples) == 0: 
                q = self.workspace.sample()
                v_nearest, dist = self.closest_vertex(q)
                #print("before",v_nearest)
                q, dist = self.steer(q, v_nearest, dist, niu, min_edge_len)
                if q != None:
                    samples.append(q)
            run = max(1,len(samples))
            while len(samples)>0:
                q = samples.pop()
                #n =self.graph.n
                #print(2 * math.sqrt((math.log(n) / n)))
                #rn = min(100/n, 5) # change rn depending on how many nodes there are, look at formula on RTT star paper

                rn = 2*max_edge_len
                neighborhood = self.near_vertices(q,rn)

                v_nearest = self.nearest(q, v_nearest, neighborhood)

                v = self.graph.add_vertex(q)
                edge = self.graph.add_edge(v,v_nearest)
                v.parent = v_nearest
                v.edge = edge

                self.re_route(v, v_nearest, neighborhood)
                new_size+=1
                if tuple(v.coord) == goal:
                    self.v_goal = v


            if self.end_condition(goal, goal_dist=1): #and i%30 == 0
                samples.append(goal)
        self.preprocessing_time = 0
        self.query_time = time.time() - start_time




    def steer(self,q, v_nearest, dist, niu, min_edge_len):
        """Steer the point closer to the tree"""
        q_nearest = v_nearest.coord
        q_new = q + (q_nearest - q) * niu / dist
        dist = dist - niu
        if self.workspace.edge_is_in_collision(q_new, q_nearest):
            # TODO
            #Idea: instead of simply returning, do something smarter?
            return None, None
        q = q_new
        if dist < min_edge_len:
            return None, None
        return tuple(q_new), dist

    def nearest(self, q, v_nearest, neighborhood):
        """Find the vertex belonging to the graph, closest to q"""
        if len(neighborhood) == 0:
            return v_nearest
        cost = self.cost(v_nearest) + self.connection(v_nearest.coord, q)
        for vertex in neighborhood:
            point = vertex.coord
            if not self.workspace.edge_is_in_collision(q,point):
                c_prime = self.cost(vertex) + self.connection(point, q)
                if cost>c_prime:
                    cost=c_prime
                    v_nearest = vertex
        return v_nearest
        
    def re_route(self, v, v_nearest, neighborhood):
        """Re route edges to minimize cost"""
        q = v.coord
        for vertex in neighborhood:
            point = vertex.coord
            if vertex is v_nearest:
                continue
            cost = self.cost(v) + self.connection(q,point)
            if self.workspace.edge_is_in_collision(q,point):
                continue
            if  (self.cost(vertex)>cost):
                self.graph.remove(vertex.edge)
                vertex.parent = v
                edge = self.graph.add_edge(v,vertex)
                vertex.edge = edge
        return v

    def end_condition(self, goal, goal_dist=0.5):
        v_nearest, dist = self.closest_vertex(goal)
        if self.v_goal!=None:
            return False
        elif dist <= goal_dist and not self.workspace.edge_is_in_collision(
            v_nearest.coord, goal
        ):
            return True 
        return False


    




def main():
    # create the workspace with some obstacles
    workspace = Workspace(20, 20)
    workspace.obstacles = [
        #Rectangle((0, -2), 1, 3),
        #Rectangle((-2, -4), 6, 1),
        #Rectangle((-2, 1), 3, 1),
        Rectangle((-5, -10), 1, 15),
        Rectangle((-1, -5), 1, 15),
        Rectangle((5, -10), 1, 15),

        #Rectangle((-4, 3), 1, 2),
        #Circle((-3, 2), 0.5),
    ]

    # start and goal locations
    start = (-8, -5)
    goal = (8, 9)
    planner = RRT_star(workspace,start)
    planner.query(start, goal,n=200,min_edge_len=0.5 ,max_edge_len=1)
    path = planner.find_path()


    print("preprocessing time:", planner.preprocessing_time)
    print("query time:", planner.query_time)
    # plot the results
    plt.figure()
    ax = plt.gca()
    workspace.draw(ax)
    planner.draw(ax)
    ax.plot(start[0], start[1], "o", color="g")
    ax.plot(goal[0], goal[1], "o", color="r")
    if path!=None:
        path.draw(ax,rgb=(0.5,1,0))
    plt.show()
    


if __name__ == "__main__":
    main()
