from environment.environment import section, Environment, Observation, Robot

if __name__ == '__main__':

    robot = Robot()
    robot.newSection()
    robot.newSection()

    env = Environment(robot)
    env.staticPoint([-75, 150])

    env.render()


    while True:
        secNum = int(input("Enter SecNum: "))
        direction = str(input("Enter direction: "))
        steps = int(input("Enter number of steps: "))

        for i in range(steps):
            obs = env.robotStep(secNum, direction)
            # print(obs)
            env.render()
