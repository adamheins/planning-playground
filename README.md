# Planning Playground

Examples of planning algorithms.

## Install
```
git clone https://github.com/adamheins/planning-playground.git
cd planning-playground

# make sure you're using Python3.7+
python3 -m pip install -r requirements.txt
python3 planning.py
```

## Usage

Currently contains a basic 2D workspace with rectangular and circular
obstacles. Planning is done between two points (start is green and goal is
red). Planners are:

* Probabilistic road map (PRM)
* Grid
