
from RRT_star.RRT_star_unbounded import RRT_star_unbounded
from src.env.Workspace import Workspace,Circle,Rectangle
from src.sampling_based_algos.RRG import RRG
from src.sampling_based_algos.RRT import RRT
from src.sampling_based_algos.Unbounded_RRT import Unbounded_RRT
from src.sampling_based_algos.Bidirectional_RRT import Bidirectional_RRT
from src.sampling_based_algos.RRT_star import RRT_star
from src.sampling_based_algos.RRT_star import RRT_star
import matplotlib.pyplot as plt

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

    # use an RRG planner
    # planner = RRG(workspace, start)

    # RRT
    #planner = RRT(workspace,start)

    # double trees
    # planner = Bidirectional_RRT(workspace,start,RRT)

    # RRT with no max distance
    # planner = Unbounded_RRT(workspace, start)

    # Unbounded bidirectional RRT
   # planner = Bidirectional_RRT(workspace, start,Unbounded_RRT)

    # RRT_star
    planner = RRT_star(workspace, start)



    planner.expand(start, goal,n=300)
    path = planner.find_path()

    # planner.add_vertices(n=100, connect_multiple_vertices=False)
    # path = planner.RRT(start,goal)
    # alternative: use a grid-based planner (only practical for small dimensions)
    # planner = Grid(workspace, 0.5)

    # alternative: use a PRM planner
    # planner = PRM(workspace)
    # planner.add_vertices(n=100)

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


