#!/usr/bin/env python

import turtle
import math
import numpy as np

turtle.setup(800, 600)
wn = turtle.Screen()
wn.tracer(300)


class baseSetup:
    def __init__(self):

        self.ground = turtle.Turtle()
        self.ground.hideturtle()

        self.points = []

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

class section:
    def __init__(self, sectionLen):
        self.section = turtle.Turtle()
        self.section.color('red')
        self.sectionLen = sectionLen
        self.zero = 0.00001

        self.leftLimit = 1.9 * math.pi
        self.currentAngle = self.zero

        self.drawSection(self.currentAngle)

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
        print(x[self.sectionLen-1] - radius, y[self.sectionLen - 1])
        self.section.home()


base = baseSetup()
base.drawGround()

arcLength = 100
sec = section(arcLength)

base.generatePoint((59.28806122141136, 70.38926642774715))

angles = np.arange(0.1, 2 * math.pi, 0.1).tolist()

while True:
    try:
        direction = str(raw_input("Enter Direction (l/r): "))
    except KeyboardInterrupt:
        exit()
    print(direction)
    if direction == 'l':
        sec.stepLeft()
    elif direction == 'r':
        sec.stepRight()
