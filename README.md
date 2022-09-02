# Planning Playground

Examples of planning algorithms for robotics.

## Install

Get the repo:
```
git clone https://github.com/adamheins/planning-playground.git
```

Install (make sure you're using **Python 3.7+**):
```
cd planning-playground
python3 -m pip install .
```

## Usage

### Planning

Currently contains a basic 2D workspace with rectangular and circular
obstacles. Planning is done between two points (start is green and goal is
red). Planners are:

* Probabilistic road map (PRM)
* Grid
* RRT
* Unbounded RRT
* Bidirectional RRT

Run the script:
```
python3 examples/all_in_one.py
```
