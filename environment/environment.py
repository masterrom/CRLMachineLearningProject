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


class section:
    def __init__(self):
        self.section = turtle.Turtle()
        self.section.color('red')
        # self.section.width(3)
        
    def drawCircle(self, radius):
        self.section.home()
        sec.section.right(-90)
        self.section.down()
        
        self.section.circle(radius)
        self.section.up()
        self.section.home()
        
        

base = baseSetup()
base.drawGround()


sec = section()
'''
Radius = 50
arcLength = 40
for i in range(100):
    t = np.linspace(0, np.math.pi/2, arcLength + 1)
    # import ipdb; ipdb.set_trace()
    x = Radius * np.cos(t)
    y = Radius * np.sin(t)
    
    
    for n in range(arcLength):
        sec.section.down()
        sec.section.setpos(x[n+1] - Radius, y[n+1])
        # sec.section.goto(x[n+1] - Radius, y[n+1])
        sec.section.up()
    sec.section.home()
    Radius = Radius + (i * 10)
'''

angles = np.arange(0.1, 2*math.pi, 0.1).tolist()
# import ipdb; ipdb.set_trace()
arcLength = 100

for i in range(l

en(angles)):
    angle = angles[i]
    radius = arcLength/angle
    
    t = np.linspace(0, angle, arcLength + 1)
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    for n in range(arcLength):
        sec.section.down()
        sec.section.setpos(x[n+1] - radius, y[n+1])
        # sec.section.goto(x[n+1] - Radius, y[n+1])
        sec.section.up()
    sec.section.home()


raw_input("Somethings")