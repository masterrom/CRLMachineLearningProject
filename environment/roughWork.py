import numpy as np
import math
import matplotlib.pyplot as plt
import turtle


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
        self.baseLocation = [x, y]

    def setBaseAngle(self, angle):
        self.baseAngle = angle

    def setRender(self, state):
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

    def drawSection(self, angle):
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

        if angle == 0:  # Use epsilon difference
            angle = self.zero

        radius = self.sectionLen / angle  # curvature is 0 ==> radius is infinite

        t = np.linspace(0, angle, self.sectionLen)
        x = radius * np.cos(t)
        y = radius * np.sin(t)

        allPoints = np.vstack((x, y)).T
        allPoints[:,0] = allPoints[:,0] - radius


        print("\t First Point", allPoints[0])
        print("\t Section Length", len(allPoints))
        print("\t Base Angle:", self.baseAngle)
        print("\t Base Location:", self.baseLocation)

        r = np.array(((np.cos(self.baseAngle), -np.sin(self.baseAngle)),
                      (np.sin(self.baseAngle), np.cos(self.baseAngle))))

        allPoints = np.dot(allPoints, r)
        allPoints[:,0] = allPoints[:,0] + self.baseLocation[0]
        allPoints[:, 1] = allPoints[:, 1] + self.baseLocation[1]

        print("\t R:",r)

        for n in range(self.sectionLen):
            px = allPoints[n][0]
            py = allPoints[n][1]
            # TODO transformation matrix previous to current configuration


            if n == 0 or n == self.sectionLen - 1:
                print('\tPoint=' + str(n) + ' :', px, py)

            # point = np.dot(r, [px, py])
            #
            #
            # px = point[0] + self.baseLocation[0]
            # py = point[1] + self.baseLocation[1]
            self.section.down()
            self.section.setpos(px, py)
            self.section.up()

        # self.tipPos = (x[len(x) - 1] - radius, y[len(y) - 1])
        self.section.home()


    def getTipPos(self):
        """
        getTipPos return a (x,y) coordinate of where the tip of the section is located
        :return: float
        """
        angle = self.currentAngle

        radius = self.sectionLen / self.currentAngle

        t = np.linspace(0, angle, self.sectionLen)
        x = radius * np.cos(t)
        y = radius * np.sin(t)

        allPoints = np.vstack((x, y)).T

        r = np.array(((np.cos(self.baseAngle), -np.sin(self.baseAngle)),
                      (np.sin(self.baseAngle), np.cos(self.baseAngle))))

        px = allPoints[-1][0] - radius + self.baseLocation[0]
        py = allPoints[-1][1] + self.baseLocation[1]
        point = np.dot(r, [px, py])

        return point

        # return (x[len(x) - 1] - radius, y[len(y) - 1])

    def displayCurve(self):
        """
        displayCurve is used of visualize all the positions that the
        section can be configured to based on its current section length
        :return: None
        """
        tool = self.curve
        tool.clear()
        tool.hideturtle()

        ang = np.linspace(-2 * math.pi, 2 * math.pi, 500)

        for i in range(len(ang)):
            angle = ang[i]
            radius = self.sectionLen / angle

            t = np.linspace(0, angle, self.sectionLen)
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            self.curve.goto(x[len(x) - 1] - radius, y[len(x) - 1])
            self.curve.dot(2, 'green')
            self.curve.up()

    def drawBaseFrame(self):
        x = self.baseFrame
        # x.clear()
        x.hideturtle()

        x.setpos(0, 0)
        x.color('red')
        x.width(3)
        x.forward(20)
        x.goto(0, 0)

        x.goto(0, 20)

    def drawEndFrame(self):
        # Based on the current angle
        end = self.endFrame
        end.reset()
        end.clear()
        end.hideturtle()
        tip = self.getTipPos()

        angle = np.rad2deg(self.currentAngle)

        end.up()
        end.setpos(tip[0], tip[1])
        end.left(0)
        end.left(90 + angle)
        end.down()
        end.forward(20)
        # print(self.currentAngle)

    def drawCircle(self):
        self.circle.up()
        self.circle.clear()
        self.circle.hideturtle()

        radius = self.sectionLen / self.currentAngle

        self.circle.setpos(-radius, -radius)
        self.circle.down()
        self.circle.circle(radius)




class Robot:
    def __init__(self):
        self.sections = []

    def newSection(self):
        newSection = section(100, 20)
        if len(self.sections) >= 1:
            newTip = self.sections[-1].getTipPos()
            newSection.setBaseLocation(newTip[0], newTip[1])
            newSection.setBaseAngle(self.sections[-1].currentAngle)
        self.sections.append(newSection)

    def step(self, secNum, action):
        secNum -= 1

        # Base Section
        self.sections[secNum].controls[action]()
        angle = self.sections[secNum].currentAngle
        tipPos = self.sections[secNum].getTipPos()
        i = secNum + 1
        while i < len(self.sections):
            self.sections[i].setBaseAngle(angle)
            self.sections[i].setBaseLocation(tipPos[0], tipPos[1])

            angle = self.sections[i].currentAngle
            tipPos = self.sections[i].getTipPos()
            i += 1

        #     self.sections[secNum+1].setBaseAngle(self.sections[secNum].currentAngle)
        #     newTip = self.sections[secNum].getTipPos()
        #     self.sections[secNum+1].setBaseLocation(newTip[0], newTip[1])
        # elif secNum == 1:
        #     self.sections[secNum].controls[action]()
        #     angle = self.sections[secNum].currentAngle
        #     tipPos = self.sections[secNum].getTipPos()
        #     self.sections[secNum + 1].setBaseAngle(angle)
        #     self.sections[secNum + 1].setBaseLocation(tipPos[0], tipPos[1])
        # elif secNum == 2:
        #     self.sections[secNum].controls[action]()

    def render(self):
        for i in range(len(self.sections)):
            sec = self.sections[i]
            sec.drawSection(sec.currentAngle)

        wn.update()




if __name__ == '__main__':
    turtle.setup(600, 800)
    wn = turtle.Screen()
    wn.delay(0)
    wn.tracer(1000)


    # sec1 = section(100, 20)
    # sec2 = section(100, 20)
    # sec2.section.color('purple')
    #
    # newTip = sec1.getTipPos()
    # sec2.setBaseLocation(newTip[0], newTip[1])
    # sec2.setBaseAngle(sec1.currentAngle)
    #
    #
    # def render(sec):
    #     points = sec.drawSection(sec.currentAngle)
    #     sec.drawEndFrame()
    #     wn.update()
    #     print('\n')
    #     return points
    #
    #
    # def drawItout(t, ps: list):
    #     t.clear()
    #     t.up()
    #     t.home()
    #     for item in ps:
    #         # t.down()
    #         t.setpos(item[0], item[1])
    #         t.dot(2, 'blue')
    #         t.up
    robot = Robot()
    robot.newSection()
    robot.newSection()
    robot.newSection()

    robot.render()


    # render(sec1)
    # render(sec2)

    while True:
        secNum = int(input('Enter section Number: '))
        dir = str(input("Direction: "))

        for i in range(10):
            robot.step(secNum, dir)
            robot.render()
            # sec1.controls[dir]()
            #
            # sec2.setBaseAngle(sec1.currentAngle)
            # newTip = sec1.getTipPos()
            # sec2.setBaseLocation(newTip[0], newTip[1])
            #
            # render(sec1)
            # points = (render(sec2))

            wn.update()
