#!/usr/bin/env python

import turtle
import math
import numpy as np
import ipdb

turtle.setup(800, 600)
wn = turtle.Screen()
wn.tracer(300)


class section:
    def __init__(self, sectionLen, maxSectionLen):
        self.section = turtle.Turtle()
        self.section.color('red')

        self.curve = turtle.Turtle()
        self.curve.color('green')

        self.sectionLen = sectionLen
        self.minSectionLen = sectionLen
        self.maxSectionLen = maxSectionLen

        self.zero = 0.00001
        self.leftLimit = 1.9 * math.pi
        self.currentAngle = self.zero
        self.tipPos = (0, 0)

        self.drawSection(self.currentAngle)
        self.displayCurve()

    def stepLeft(self):
        # increase by angle by .5 degree
        if self.currentAngle == self.zero:
            self.currentAngle = 0

        self.currentAngle += 0.1
        if self.currentAngle == self.leftLimit:
            self.currentAngle -= 0.1
            print("Left angle limit reached", self.currentAngle)
            return

        self.drawSection(self.currentAngle)

    def stepRight(self):
        # decrease by angle by .5 degree
        if self.currentAngle == self.zero:
            self.currentAngle = 0

        self.currentAngle -= 0.1
        if self.currentAngle == -self.leftLimit:
            self.currentAngle += 0.1
            print("Right angle limit reached", self.currentAngle)
            return

        self.drawSection(self.currentAngle)

    def extendArm(self):
        if self.sectionLen == self.maxSectionLen:
            assert "Section max length has reached"
            return
        else:
            self.sectionLen += 1
        self.drawSection(self.currentAngle)
        self.displayCurve()
        return

    def contractArm(self):
        if self.sectionLen == self.minSectionLen:
            assert "Section min length has reached"
            return
        else:
            self.sectionLen -= 1
        self.drawSection(self.currentAngle)
        self.displayCurve()
        return

    def drawSection(self, angle):

        self.section.clear()
        self.section.hideturtle()
        if angle == 0:
            angle = self.zero

        radius = self.sectionLen / angle

        t = np.linspace(0, angle, self.sectionLen)
        x = radius * np.cos(t)
        y = radius * np.sin(t)

        for n in range(self.sectionLen):
            self.section.down()
            self.section.setpos(x[n] - radius, y[n])
            self.section.up()

        self.tipPos = (x[len(x) - 1] - radius, y[len(y) - 1])
        self.section.home()

    def getTipPos(self):
        return self.tipPos

    def displayCurve(self):

        tool = self.curve
        tool.clear()
        tool.hideturtle()

        ang = np.linspace(0, 2 * math.pi, 500)
        negAng = np.linspace(0, -2 * math.pi, 500)

        for i in range(len(ang)):
            angle = ang[i]
            radius = self.sectionLen / angle

            t = np.linspace(0, angle, self.sectionLen)
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            self.curve.goto(x[len(x) - 1] - radius, y[len(x) - 1])
            self.curve.dot(2, 'green')
            self.curve.up()

            angle = negAng[i]
            radius = self.sectionLen / angle

            t = np.linspace(0, angle, self.sectionLen)
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            self.curve.goto(x[len(x) - 1] - radius, y[len(x) - 1])
            self.curve.dot(2, 'green')


class Environment:
    def __init__(self, robot: section):

        self.ground = turtle.Turtle()
        self.ground.hideturtle()
        self.taskSpace = {'dim': (100, 100),
                          'tool': turtle.Turtle()}
        self.taskSpace['tool'].hideturtle()
        self.robot = robot
        self.capPoints = 0
        self.points = []

    def drawTaskSpace(self):
        # global wn
        #
        # wn.tracer(100)

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

        global wn

        wn.tracer(600)

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

    def pointCapture(self):
        tipPos = self.robot.getTipPos()

        pCap = None
        for i in range(len(self.points)):
            # Get Distance

            pos = self.points[i].pos()

            a = pos[0] - tipPos[0]
            b = pos[1] - tipPos[1]

            c = math.sqrt(a ** 2 + b ** 2)
            if c <= 0.05:
                self.capPoints += 1
                pCap = i
                print("Captured Point: Points ", self.capPoints)
                break
        if pCap is not None:
            self.points[pCap].clear()
            self.points.pop(pCap)

    def generatePoint(self, p):
        point = turtle.Turtle()
        point.hideturtle()
        point.up()

        point.setpos(p[0], p[1])
        point.color('blue')
        point.down()
        point.dot(4)
        self.points.append(point)
        point.up()


arcLength = 100
robot = section(arcLength, 120)

base = Environment(robot)
base.drawGround()

base.generatePoint((59.28806122141136, 70.38926642774715))
print(base.points)
angles = np.arange(0.1, 2 * math.pi, 0.1).tolist()

while True:
    print("l - move left | r - move right | e - extend | c - contract")
    try:
        direction = str(input("Enter Direction (l/r/e/c): "))
    except KeyboardInterrupt:
        exit()
    print(direction)
    if direction == 'l':
        robot.stepLeft()
    elif direction == 'r':
        robot.stepRight()
    elif direction == 'e':
        robot.extendArm()
    elif direction == 'c':
        robot.contractArm()

    base.pointCapture()
