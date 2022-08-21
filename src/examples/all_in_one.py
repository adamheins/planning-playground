
from src.env.Workspace import Workspace,Circle,Rectangle
from src.sampling_based_algos.RRG import RRG
from src.sampling_based_algos.RRT import RRT
from src.sampling_based_algos.Bidirectional_RRT import Bidirectional_RRT
from src.sampling_based_algos.RRT_star import RRT_star

import matplotlib.pyplot as plt
import time

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
    t = time.time()
    # use an RRG planner
    # planner = RRG(workspace, start)

    # RRT
    # planner = RRT(workspace,start)
    # planner.extend( goal, n=200, min_edge_len=0.5, max_edge_len=1, niu=1, divide_edges=False, stop_early=True)

    # double trees
    # planner = Bidirectional_RRT(workspace,start,RRT)
    # planner.extend(goal, n=150, min_edge_len=0.5, max_edge_len=1, niu=1, divide_edges=False, stop_early=True)

    # Unbounded RRT
    # planner = RRT(workspace, start)
    # planner.extend( goal, n=250, min_edge_len=0.5, max_edge_len=1, niu=1, divide_edges=True, stop_early=True)

    # Unbounded bidirectional RRT
    # planner = Bidirectional_RRT(workspace, start, RRT)
    # planner.extend( goal, n=170, min_edge_len=0.5, max_edge_len=1, niu=1, divide_edges=True, stop_early=True)

    # RRT_star
    # planner = RRT_star(workspace, start)
    # planner.extend( goal, n=150, min_edge_len=0.5, max_edge_len=1, niu=1, divide_edges=False, stop_early=False)

    # Unbounded RRT_star
    # planner = RRT_star(workspace, start)
    # planner.extend( goal, n=150, min_edge_len=0.5, max_edge_len=1, niu=1, divide_edges=True, stop_early=False)

    # bidirectional RRT_star
    planner = Bidirectional_RRT(workspace, start, RRT_star)
    planner.extend( goal, n=170, min_edge_len=0.5, max_edge_len=1, niu=1, divide_edges=True, stop_early=True)






    # while not planner.extend(goal, n=100, min_edge_len=0.5, max_edge_len=5):
    #     pass
    #planner.extend( goal, n=100, min_edge_len=0.5, max_edge_len=1, niu=1, divide_edges=True, stop_early=True)
    query_time = time.time() - t

    path = planner.get_path_to_goal()

    print(f"query time: {query_time}")
    

    # plot the results
    plt.figure()
    ax = plt.gca()
    workspace.draw(ax)
    planner.draw(ax)
    ax.plot(start[0], start[1], "o", color="g")
    ax.plot(goal[0], goal[1], "o", color="r")
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color="g")
    plt.show()


if __name__ == "__main__":
    main()


