# graphicsCrawlerDisplay.py
# -------------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: This adaptation is based on AI projects developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# graphicsCrawlerDisplay.py
# -------------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The original AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://ai.berkeley.edu.

import tkinter, math
from math import pi as PI

robotType = 'crawler'

class CrawlerRenderer:
    def __init__(self, canvas, robot):
        self.robot = robot
        self.canvas = canvas
        self.velAvg = 0
        self.lastStep = 0
        self.ground = canvas.create_rectangle(0,
            self.robot.groundY,self.robot.totWidth,self.robot.totHeight, fill='blue')
        self.robotBody = canvas.create_polygon(0,0,0,0,0,0,0,0, fill='green')
        self.robotArm = canvas.create_line(0,0,0,0,fill='orange',width=5)
        self.robotHand = canvas.create_line(0,0,0,0,fill='red',width=3)

    def __getCosAndSin(self, angle):
        return math.cos(angle), math.sin(angle)

    def draw(self, stepCount, stepDelay):
        x1, y1 = self.robot.getRobotPosition()
        x1 = x1 % self.robot.totWidth
        if y1 != self.robot.groundY:
            raise Exception('Flying Robot!!')
        rotationAngle = self.robot.getRotationAngle()
        cosRot, sinRot = self.__getCosAndSin(rotationAngle)
        x2 = x1 + self.robot.robotWidth * cosRot
        y2 = y1 - self.robot.robotWidth * sinRot
        x3 = x1 - self.robot.robotHeight * sinRot
        y3 = y1 - self.robot.robotHeight * cosRot
        x4 = x3 + cosRot * self.robot.robotWidth
        y4 = y3 - sinRot * self.robot.robotWidth
        self.canvas.coords(self.robotBody, x1, y1, x2, y2, x4, y4, x3, y3)
        armCos, armSin = self.__getCosAndSin(rotationAngle + self.robot.armAngle)
        xArm = x4 + self.robot.armLength * armCos
        yArm = y4 - self.robot.armLength * armSin
        self.canvas.coords(self.robotArm, x4, y4, xArm, yArm)
        handCos, handSin = self.__getCosAndSin(self.robot.handAngle + rotationAngle)
        xHand = xArm + self.robot.handLength * handCos
        yHand = yArm - self.robot.handLength * handSin
        self.canvas.coords(self.robotHand, xArm, yArm, xHand, yHand)
        steps = stepCount - self.lastStep
        if steps == 0:
            return
        pos = self.robot.positions[-1]
        velocity = pos - self.robot.positions[-2]
        vel2 = (pos - self.robot.positions[0]) / len(self.robot.positions)
        self.velAvg = 0.9 * self.velAvg + 0.1 * vel2
        velMsg = '100-step Avg Velocity: %.2f' % self.velAvg
        velocityMsg = 'Velocity: %.2f' % velocity
        positionMsg = 'Position: %2.f' % pos
        stepMsg = 'Step: %d' % stepCount
        if 'vel_msg' in dir(self):
            self.canvas.delete(self.vel_msg)
            self.canvas.delete(self.pos_msg)
            self.canvas.delete(self.step_msg)
            self.canvas.delete(self.velavg_msg)
        self.velavg_msg = self.canvas.create_text(650, 190, text=velMsg)
        self.vel_msg = self.canvas.create_text(450, 190, text=velocityMsg)
        self.pos_msg = self.canvas.create_text(250, 190, text=positionMsg)
        self.step_msg = self.canvas.create_text(50, 190, text=stepMsg)
        self.lastStep = stepCount


class CrawlerControls:
    def sigmoid(self, x):
        return 1.0 / (1.0 + 2.0 ** (-x))

    def incrementSpeed(self, inc):
        self.tickTime *= inc
    #        self.epsilon = min(1.0, self.epsilon)
    #        self.epsilon = max(0.0,self.epsilon)
    #        self.learner.setSpeed(self.epsilon)
        self.speed_label['text'] = 'Step Delay: %.5f' % (self.tickTime)

    def incrementEpsilon(self, inc):
        self.ep += inc
        self.epsilon = self.sigmoid(self.ep)
        self.learner.setEpsilon(self.epsilon)
        self.epsilon_label['text'] = 'Epsilon: %.3f' % (self.epsilon)

    def incrementGamma(self, inc):
        self.ga += inc
        self.gamma = self.sigmoid(self.ga)
        self.learner.setDiscount(self.gamma)
        self.gamma_label['text'] = 'Discount: %.3f' % (self.gamma)

    def incrementAlpha(self, inc):
        self.al += inc
        self.alpha = self.sigmoid(self.al)
        self.learner.setLearningRate(self.alpha)
        self.alpha_label['text'] = 'Learning Rate: %.3f' % (self.alpha)

    def initialize_gui(self, win):
        ## Window ##
        self.win = win

        ## Initialize Frame ##
        win.grid()
        self.dec = -.5
        self.inc = .5
        self.tickTime = 0.1

        ## Epsilon Button + Label ##
        self.setupSpeedButtonAndLabel(win)

        self.setupEpsilonButtonAndLabel(win)

        ## Gamma Button + Label ##
        self.setUpGammaButtonAndLabel(win)

        ## Alpha Button + Label ##
        self.setupAlphaButtonAndLabel(win)

        ## Exit Button ##
        #self.exit_button = tkinter.Button(win,text='Quit', command=self.exit)
        #self.exit_button.grid(row=0, column=9)

        ## Simulation Buttons ##
    #        self.setupSimulationButtons(win)

         ## Canvas ##
        self.canvas = tkinter.Canvas(win, height=200, width=1000)
        self.canvas.grid(row=2,columnspan=10)

    def setupAlphaButtonAndLabel(self, win):
        self.alpha_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementAlpha(self.dec)))
        self.alpha_minus.grid(row=1, column=3, padx=10)

        self.alpha = self.sigmoid(self.al)
        self.alpha_label = tkinter.Label(win, text='Learning Rate: %.3f' % (self.alpha))
        self.alpha_label.grid(row=1, column=4)

        self.alpha_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementAlpha(self.inc)))
        self.alpha_plus.grid(row=1, column=5, padx=10)

    def setUpGammaButtonAndLabel(self, win):
        self.gamma_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementGamma(self.dec)))
        self.gamma_minus.grid(row=1, column=0, padx=10)

        self.gamma = self.sigmoid(self.ga)
        self.gamma_label = tkinter.Label(win, text='Discount: %.3f' % (self.gamma))
        self.gamma_label.grid(row=1, column=1)

        self.gamma_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementGamma(self.inc)))
        self.gamma_plus.grid(row=1, column=2, padx=10)

    def setupEpsilonButtonAndLabel(self, win):
        self.epsilon_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementEpsilon(self.dec)))
        self.epsilon_minus.grid(row=0, column=3)

        self.epsilon = self.sigmoid(self.ep)
        self.epsilon_label = tkinter.Label(win, text='Epsilon: %.3f' % (self.epsilon))
        self.epsilon_label.grid(row=0, column=4)

        self.epsilon_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementEpsilon(self.inc)))
        self.epsilon_plus.grid(row=0, column=5)

    def setupSpeedButtonAndLabel(self, win):
        self.speed_minus = tkinter.Button(win,
        text="-",command=(lambda: self.incrementSpeed(.5)))
        self.speed_minus.grid(row=0, column=0)

        self.speed_label = tkinter.Label(win, text='Step Delay: %.5f' % (self.tickTime))
        self.speed_label.grid(row=0, column=1)

        self.speed_plus = tkinter.Button(win,
        text="+",command=(lambda: self.incrementSpeed(2)))
        self.speed_plus.grid(row=0, column=2)

    def skip5kSteps(self):
        self.stepsToSkip = 5000

    def animatePolicy(self):
        if robotType != 'pendulum':
            raise Exception('Only pendulum can animatePolicy')


        totWidth = self.canvas.winfo_reqwidth()
        totHeight = self.canvas.winfo_reqheight()

        length = 0.48 * min(totWidth, totHeight)
        x,y = totWidth-length-30, length+10



        angleMin, angleMax = self.robot.getMinAndMaxAngle()
        velMin, velMax = self.robot.getMinAndMaxAngleVelocity()

        if not 'animatePolicyBox' in dir(self):
            self.canvas.create_line(x,y,x+length,y)
            self.canvas.create_line(x+length,y,x+length,y-length)
            self.canvas.create_line(x+length,y-length,x,y-length)
            self.canvas.create_line(x,y-length,x,y)
            self.animatePolicyBox = 1
            self.canvas.create_text(x+length/2,y+10,text='angle')
            self.canvas.create_text(x-30,y-length/2,text='velocity')
            self.canvas.create_text(x-60,y-length/4,text='Blue = kickLeft')
            self.canvas.create_text(x-60,y-length/4+20,text='Red = kickRight')
            self.canvas.create_text(x-60,y-length/4+40,text='White = doNothing')



        angleDelta = (angleMax-angleMin) / 100
        velDelta = (velMax-velMin) / 100
        for i in range(100):
            angle = angleMin + i * angleDelta

            for j in range(100):
                vel = velMin + j * velDelta
                state = self.robotEnvironment.getState(angle,vel)
                max, argMax = None, None
                if not self.learner.seenState(state):
                    argMax = 'unseen'
                else:
                    for action in ('kickLeft','kickRight','doNothing'):
                        qVal = self.learner.getQValue(state, action)
                        if max == None or qVal > max:
                            max, argMax = qVal, action
                if argMax != 'unseen':
                    if argMax == 'kickLeft':
                        color = 'blue'
                    elif argMax == 'kickRight':
                        color = 'red'
                    elif argMax == 'doNothing':
                        color = 'white'
                    dx = length / 100.0
                    dy = length / 100.0
                    x0, y0 = x+i*dx, y-j*dy
                    self.canvas.create_rectangle(x0,y0,x0+dx,y0+dy,fill=color)

    def start(self):
        self.win.mainloop()
