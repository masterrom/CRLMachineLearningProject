# CRLMachineLearningProject

Play with the robot by running `./environmentTest.py`

## Documentation
- Simply Open `index.html` for all documentation

## Current Setup
- Multi-Section Configurable
```
robot = Robot() # New Robot instance
robot.newSection() # Add a new Section
robot.newSection() # Add a second Section

env = Environment(robot) # Add robot to the environment
```
- Available Actions for each sections
```
l - increase curvature
r - decrease curvature
e - increase section length
c - decrease section length
```
- Curvature : [2pi, -2pi]
- sectionLen: [100, 120]

- Curvature StepSize: 0.01
- sectionLen StepSize: 1
 
## Models
- `Model.py` = DDQN
- `a2c.py` = ActorCritic Network

## Environment
- `environment/environment.py`
