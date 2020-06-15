# CRLMachineLearningProject

Play with the robot by running `./environment.py` in the `environment Folder`

Actions:
- l = Curve towards the left
- r = Curve towards the right
- e = extend the robot ( A little glitchy )
- c = contract the robot ( glitchy )

## Documentation
- run `pdoc environment.py --html --force` to compile documents
- run  `pdoc environment.py --http localhost:3000` to host or can simply open up the html file

## Current Setup
- Environment Holds a single Section Robot
- Each step is taken through the environment
    - Before taking each step, the previous state is saved
 
## TODO
- Implement basic Model
