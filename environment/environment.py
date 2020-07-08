#!/usr/bin/env python
import time
import turtle
import math
import random
from typing import Any

from dataclasses import dataclass
import numpy as np
from numpy.random import randint, seed
import ipdb


@dataclass
class Observation:
    state: Any
    nextState: Any
    reward: float
    action: tuple


def distance(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return math.sqrt(x ** 2 + y ** 2)


class section:

    def __init__(self, sectionLen, maxSectionLen):
        """
        Section class represents a single section of a continuum robot

        :param sectionLen: Minimum section length\n
        :param maxSectionLen: Maximum Section length
        """

        self.baseLocation = [0, 0]
        self.baseAngle = 0
        self.transformations = []
        # Set of baseLocations and Angles

        self.section = turtle.Turtle()
        self.section.color('red')

        self.render = False

        self.baseFrame = turtle.Turtle()
        self.endFrame = turtle.Turtle()

        self.curve = turtle.Turtle()
        self.curve.color('green')

        self.circle = turtle.Turtle()
        self.circle.color('pink')

        self.sectionLen = sectionLen
        self.minSectionLen = sectionLen
        self.maxSectionLen = maxSectionLen

        self.zero = 0.00001
        self.leftLimit = 1.9 * math.pi
        self.currentAngle = self.zero
        self.tipPos = (0, 0)

        # self.drawSection(self.currentAngle)
        # self.displayCurve()

        self.controls = {
            'l': self.stepLeft,
            'r': self.stepRight,
            'e': self.extendArm,
            'c': self.contractArm
        }
        self.controlNum = {
            'l': 0, 'r': 1, 'e': 2, 'c': 3
        }

    def setBaseLocation(self, x, y):
        '''
        setBaseLocation is used modify the starting position of
        the section.
        :param x: x coordinate of the new base position
        :param y: y coordinate of the new base position
        :return: None
        '''
        self.baseLocation = [x, y]

    def setRender(self, state):
        '''
        setRender Method is used enable/disable rendering
        :param state: bool
        :return: None
        '''
        self.render = state

    def stepLeft(self):
        """
        stepLeft function will increase the curvature angle towards 2π
        while the section Length will remain the same. Each step will
        increase the angle by 0.01
        :return: None
        """
        # increase by angle by .5 degree
        if self.currentAngle == self.zero:
            self.currentAngle = 0

        self.currentAngle += 0.01
        if self.currentAngle == self.leftLimit:
            self.currentAngle -= 0.01
            print("Left angle limit reached", self.currentAngle)
            return

        if self.render:
            self.drawSection(self.currentAngle)

    def stepRight(self):
        """
        stepRight function will decrease the curvature angle towards -2π
        while the section Length will remain the same. Each step will
        decrease the angle by 0.01
        :return: None
        """
        # decrease by angle by .5 degree
        if self.currentAngle == self.zero:
            self.currentAngle = 0

        self.currentAngle -= 0.01
        if self.currentAngle == -self.leftLimit:
            self.currentAngle += 0.01
            print("Right angle limit reached", self.currentAngle)
            return
        if self.render:
            self.drawSection(self.currentAngle)

    def extendArm(self):
        """
        extendArm will increase the length of section by a single unit
        if the length is below the maximum increase. Curvature will remain
        constant
        :return: None
        """
        if self.sectionLen == self.maxSectionLen:
            assert "Section max length has reached"
            return
        else:
            self.sectionLen += 1

        if self.render:
            self.drawSection(self.currentAngle)
            self.displayCurve()
        return

    def contractArm(self):
        """
        contractArm will decrease the length of section by a single unit
        if the length is above the minimum length. Curvature will remain
        constant
        :return: None
        """
        if self.sectionLen == self.minSectionLen:
            assert "Section min length has reached"
            return
        else:
            self.sectionLen -= 1

        if self.render:
            self.drawSection(self.currentAngle)
            self.displayCurve()

        return

    def drawSection(self, transformations):
        """
        drawSection will draw the current representation of the section
        based on the given angle and current section length
        :param angle: Angle of the curvature. Should not be used
        :return: None
        """
        self.section.clear()
        self.section.hideturtle()
        self.section.up()
        self.section.setpos(self.baseLocation[0], self.baseLocation[1])

        angle = self.currentAngle
        if angle == 0:  # Use epsilon difference
            angle = self.zero

        radius = self.sectionLen / angle  # curvature is 0 ==> radius is infinite

        t = np.linspace(0, angle, self.sectionLen)
        x = radius * np.cos(t)
        y = radius * np.sin(t)

        allPoints = np.vstack((x, y)).T
        allPoints[:,0] = allPoints[:,0] - radius


        # print("\t First Point", allPoints[0])
        # print("\t Base Location:", self.baseLocation)

        for i in range(len(transformations)):
            baseAngle = transformations[i]
            r = np.array(((np.cos(baseAngle), -np.sin(baseAngle)),
                          (np.sin(baseAngle), np.cos(baseAngle))))
            allPoints = np.dot(r, allPoints.T).T

        allPoints[:,0] = allPoints[:,0] + self.baseLocation[0]
        allPoints[:, 1] = allPoints[:, 1] + self.baseLocation[1]


        for n in range(self.sectionLen):
            px = allPoints[n][0]
            py = allPoints[n][1]

            # if n == 0 or n == self.sectionLen - 1:
            #     print('\tPoint=' + str(n) + ' :', px, py)

            self.section.down()
            self.section.setpos(px, py)
            self.section.up()

        # self.tipPos = (x[len(x) - 1] - radius, y[len(y) - 1])
        self.section.home()

    def getTipPos(self, transformations):
        """
        getTipPos return a (x,y) coordinate of where the tip of the section is located
        :return: float
        """
        angle = self.currentAngle
        if angle == 0:  # Use epsilon difference
            angle = self.zero

        radius = self.sectionLen / angle  # curvature is 0 ==> radius is infinite

        t = np.linspace(0, angle, self.sectionLen)
        x = radius * np.cos(t)
        y = radius * np.sin(t)

        allPoints = np.vstack((x, y)).T
        allPoints[:, 0] = allPoints[:, 0] - radius

        for i in range(len(transformations)):
            baseAngle = transformations[i]
            r = np.array(((np.cos(baseAngle), -np.sin(baseAngle)),
                          (np.sin(baseAngle), np.cos(baseAngle))))
            allPoints = np.dot(r, allPoints.T).T

        allPoints[:, 0] = allPoints[:, 0] + self.baseLocation[0]
        allPoints[:, 1] = allPoints[:, 1] + self.baseLocation[1]

        px = allPoints[-1][0]
        py = allPoints[-1][1]

        point = [px, py]

        return point

    def displayCurve(self, transformations):
        """
        displayCurve is used of visualize all the positions that the
        section can be configured to based on its current section length
        :return: None
        """
        tool = self.curve
        tool.clear()
        tool.hideturtle()
        tool.up()

        ang = np.linspace(-2 * math.pi, 2 * math.pi, 500)

        for i in range(len(ang)):
            angle = ang[i]
            radius = self.sectionLen / angle

            t = np.linspace(0, angle, self.sectionLen)
            x = radius * np.cos(t)
            y = radius * np.sin(t)

            x = x[-1]
            y = y[-1]

            x = x - radius
            points = [x,y]
            for i in range(len(transformations)):
                baseAngle = transformations[i]
                r = np.array(((np.cos(baseAngle), -np.sin(baseAngle)),
                              (np.sin(baseAngle), np.cos(baseAngle))))
                points = np.dot(r, points)

            points[0] = points[0] + self.baseLocation[0]
            points[1] = points[1] + self.baseLocation[1]

            self.curve.goto(points[0], points[1])
            self.curve.dot(2, 'green')
            self.curve.up()


class Robot:
    def __init__(self):
        '''
        Robot class is a super class, which can be used to
        make a series of sections to work together.
        '''
        self.sections = []
        self.zero = 0.00001
        self.tipPosition = 0
        self.controlNum = {
            'l': 0, 'r': 1, 'e': 2, 'c': 3
        }

    def newSection(self):
        '''
        newSection creates a new section and adds it to the end of
        the robot. Default section is 100
        :return: None
        '''
        newSection = section(100, 20)
        angles = self.getAllCurrentAngles()
        if len(self.sections) >= 1:
            newTip = self.sections[-1].getTipPos(angles)
            newSection.setBaseLocation(newTip[0], newTip[1])
        self.sections.append(newSection)

    def getAllCurrentAngles(self):
        '''
        getAllCurrentAngles gets the curvature angle for each section
        :return: list[float]
        '''
        angles = []
        for i in range(len(self.sections)):
            angle = self.sections[i].currentAngle
            # if angle != self.zero:
            angles.append(angle)

        return angles

    def getAllSectionConfigurations(self):
        angles = self.getAllCurrentAngles()
        configs = []
        for i in range(len(self.sections)):
            angle = angles[i]
            secLen = self.sections[i].sectionLen
            config = (angle, secLen)
            configs.append(config)

        return configs

    def step(self, secNum, action):
        '''
        step methods will conduct the given action for the given section
        :param secNum: Section number in the robot (index starting at 1)
        :param action: action (l,r,c,e)
        :return:  None
        '''
        secNum -= 1
        # # Base Section

        self.sections[secNum].controls[action]()

        angles = self.getAllCurrentAngles()

        tipPos = self.sections[secNum].getTipPos(angles[:secNum])
        i = secNum + 1
        while i < len(self.sections):

            self.sections[i].setBaseLocation(tipPos[0], tipPos[1])

            tipPos = self.sections[i].getTipPos(angles[:i])
            i += 1

    def endEffectorPos(self):
        lastSection = self.sections[len(self.sections) - 1]
        transformations = self.getAllCurrentAngles()
        tipPos = lastSection.getTipPos(transformations[:len(self.sections) - 1])

        return tipPos

    def render(self):
        '''
        render will draw out each section
        :return:
        '''
        angles = self.getAllCurrentAngles()
        for i in range(len(self.sections)):
            # print("--------- Section " + str(i) + '-----------')
            sec = self.sections[i]
            transformations = angles[:i]
            # print('\t transformations:', transformations)
            sec.drawSection(transformations)
            # sec.displayCurve(transformations)


class Environment:

    def __init__(self, robot: Robot):
        """
        Environment class holds the entire game. Where the
        robot(made out of several sections) is the player. And the
        purpose of the game is to capture as many points as possible
        without hitting any obstacles in between
        :param robot: Section
        """
        turtle.setup(800, 600)
        self.wn = None
        self.ground = turtle.Turtle()
        self.ground.hideturtle()
        self.taskSpace = {'dim': (100, 100),
                          'tool': turtle.Turtle()}

        self.checkTipPoint = turtle.Turtle()
        self.checkTipPoint.hideturtle()


        self.taskSpace['tool'].hideturtle()
        self.robot = robot
        self.capPoints = 0
        self.points = []
        self.generatePoint()

        self.rewardTool = turtle.Turtle()
        self.rewardTool.hideturtle()
        # self.drawReward()

        self.prevState = []
        self.prevState.extend(robot.getAllSectionConfigurations())
        self.prevState.append(distance(self.points[0], robot.endEffectorPos()))

        # self.prevState = [self.robot.currentAngle,
        #                   self.robot.sectionLen,
        #                   distance(self.robot.getTipPos(),
        #                            (self.points[0][0], self.points[0][1]))]  # Arc Parameters - Distance to Point
        #

        self.currentState = self.prevState

        self.observation = Observation(self.prevState, self.prevState, -self.prevState[2], (1,0))

    def drawTaskSpace(self):
        """
        drawTaskSpace is not in use yet
        :return:
        """

        tool = self.taskSpace['tool']

        def makeLine(length):
            for i in range(length):
                tool.forward(1)

        tool.up()
        tool.color("green")
        tool.home()
        tool.width(7)

        tool.down()
        makeLine(100)

        tool.up()
        tool.home()
        tool.right(180)
        tool.down()
        makeLine(100)

        tool.up()
        tool.right(90)
        tool.down()
        makeLine(130)

        tool.up()
        tool.right(90)
        tool.down()
        makeLine(200)

    def drawGround(self):
        """
        drawGround will draw the initial base line of the game
        currently holds no physics
        :return: None
        """
        self.wn.tracer(600)

        self.ground.color("black")
        self.ground.down()

        for i in range(300):
            self.ground.dot()
            self.ground.forward(1)

        self.ground.home()
        self.ground.right(-180)

        self.ground.down()
        for i in range(300):
            self.ground.dot()
            self.ground.forward(1)

        self.ground.up()
        self.ground.home()

    def drawReward(self):
        """
        drawReward will draws out the current reward counter
        (ie: how many points the robot has been able to capture)
        :return:
        """
        self.rewardTool.clear()
        self.rewardTool.color('black')
        self.rewardTool.up()
        self.rewardTool.setpos(0, -50)
        self.rewardTool.down()
        self.rewardTool.write("Points: " + str(self.capPoints), align='center')

    def render(self):
        if self.wn is None:
            self.wn = turtle.Screen()
            # self.wn.delay(0)
            # self.wn.tracer(600)
            self.drawGround()
            # self.robot.setRender(True)

            # self.robot.drawSection(self.robot.currentAngle)
            # self.robot.displayCurve()

        self.robot.render()

        self.drawReward()
        turtle.Screen().update()
        self.drawPoint()

    def pointCapture(self):
        """
        pointCapture checks if robot tipPosition is within a
        0.5 radius of the un-captured Point. if so, accumulate
        the points and generate a new random points
        :return: None
        """
        tipPos = self.robot.endEffectorPos()
        self.checkTipPoint.clear()
        self.checkTipPoint.up()

        print("in here")
        print("tipPos: ", tipPos)
        self.checkTipPoint.setpos(tipPos[0], tipPos[1])
        self.checkTipPoint.dot(3,'blue')
        pCap = None
        for i in range(len(self.points)):
            # Get Distance

            pos = self.points[i]

            c = distance(pos, tipPos)

            print("C Val: ", c)
            if c <= 2:
                self.capPoints += 1
                pCap = i
                print("Captured Point: Points ", self.capPoints)
                break
        if pCap is not None:
            self.points[pCap].clear()
            self.points.pop(pCap)
            self.generatePoint()
            return True
        return False

    def getObservation(self):
        # Current State, reward
        return

    def generatePoint(self):
        """
        generatePoint generates a random point based on the
        robot maxSection - minSection and maxCurvature - minCurvature
        :return: (x, y)
        """
        maxHeight = len(self.robot.sections) * 100
        maxWidth = 100

        x = random.uniform(-maxWidth, maxWidth)
        y = random.uniform(10, maxHeight)
        point = [x,y]
        self.points.append(point)


        return [x, y]
        # angle = random.uniform(-2 * math.pi, 2 * math.pi)
        # maxArcLen = self.robot.maxSectionLen
        # minArcLen = self.robot.minSectionLen
        #
        # arcLen = random.randint(minArcLen, maxArcLen)
        #
        # radius = arcLen / angle
        #
        # t = np.linspace(0, angle, arcLen)
        # x = radius * np.cos(t)
        # y = radius * np.sin(t)
        #
        # point = [x[arcLen - 1] - radius, y[arcLen - 1]]
        # self.points.append(point)

    def drawPoint(self):
        """
        drawPoint generates a random point, draws a point on board
        :return: None
        """
        p = self.points[0]
        point = turtle.Turtle()
        point.hideturtle()
        point.up()

        point.setpos(p[0], p[1])
        point.color('blue')
        point.down()
        point.dot(4)
        point.up()

    def robotStep(self, sec, direction):
        """
        robotStep will take a step towards the specified direction
        :param direction: integer (0,1,2,3)
        :return: Observation
        """

        # self.robot.step(sec, direction)


        robot = self.robot
        # Save previous state
        self.prevState = self.currentState

        # Step
        self.robot.step(sec, direction)
        self.currentState = []
        self.currentState.extend(robot.getAllSectionConfigurations())
        self.currentState.append(distance(self.points[0], robot.endEffectorPos()))


        reward = -self.currentState[-1]

        # Determine if a point was captured
        capPoint = self.pointCapture()
        if capPoint:
            reward += 100

        self.observation = Observation(self.prevState, self.currentState, reward, (sec, self.robot.controlNum[direction]))
        print(self.observation)
        return self.observation


    def randomAction(self):
        """
        randomAction will generate a random action
        :return: int
        """
        # TODO add in random seed
        action = randint(0, 3)
        return action
