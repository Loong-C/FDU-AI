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

import tkinter, time, threading, traceback, sys, math
from math import pi as PI
from Agents.Coursework import qlearning_agents as qlearningAgents
from Game.Core import crawler
from Game.Presentation.crawler_display import CrawlerControls, CrawlerRenderer

robotType = 'crawler'

class Application(CrawlerControls):
    def __init__(self, win):

        self.ep = 0
        self.ga = 2
        self.al = 2
        self.stepCount = 0
        ## Init Gui

        self.initialize_gui(win)

        # Init environment
        if robotType == 'crawler':
            self.robot = crawler.CrawlingRobot(self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight())
            self.renderer = CrawlerRenderer(self.canvas, self.robot)
            self.robotEnvironment = crawler.CrawlingRobotEnvironment(self.robot)
        elif robotType == 'pendulum':
            self.robot = pendulum.PendulumRobot(self.canvas)
            self.robotEnvironment = \
                pendulum.PendulumRobotEnvironment(self.robot)
        else:
            raise Exception("Unknown RobotType")

        # Init Agent
        simulationFn = lambda agent: \
          simulation.SimulationEnvironment(self.robotEnvironment,agent)
        actionFn = lambda state: \
          self.robotEnvironment.getPossibleActions(state)
        self.learner = qlearningAgents.QLearningAgent(actionFn=actionFn)

        self.learner.setEpsilon(self.epsilon)
        self.learner.setLearningRate(self.alpha)
        self.learner.setDiscount(self.gamma)

        # Start GUI
        self.running = True
        self.stopped = False
        self.stepsToSkip = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def exit(self):
        self.running = False
        for i in range(5):
            if not self.stopped:
                time.sleep(0.1)
        try:
            self.win.destroy()
        except:
            pass
        sys.exit(0)

    def step(self):

        self.stepCount += 1

        state = self.robotEnvironment.getCurrentState()
        actions = self.robotEnvironment.getPossibleActions(state)
        if len(actions) == 0.0:
            self.robotEnvironment.reset()
            state = self.robotEnvironment.getCurrentState()
            actions = self.robotEnvironment.getPossibleActions(state)
            print('Reset!')
        action = self.learner.getAction(state)
        if action == None:
            raise Exception('None action returned: Code Not Complete')
        nextState, reward = self.robotEnvironment.doAction(action)
        self.learner.observeTransition(state, action, nextState, reward)

    def run(self):
        self.stepCount = 0
        self.learner.startEpisode()
        try:
            while self.running:
                delay = max(0.01, self.tickTime)
                time.sleep(delay)
                self.stepsToSkip = int(delay / self.tickTime) - 1
                if not self.running:
                    break
                for _ in range(self.stepsToSkip):
                    self.step()
                self.stepsToSkip = 0
                self.step()
        except (Exception, SystemExit):
            traceback.print_exc()
            self.running = False
        finally:
            self.learner.stopEpisode()
            self.stopped = True


def run():
    global root
    root = tkinter.Tk()
    root.title( 'Crawler GUI' )
    root.resizable( 0, 0 )

#  root.mainloop()


    app = Application(root)
    def update_gui():
        app.renderer.draw(app.stepCount, app.tickTime)
        root.after(10, update_gui)
    update_gui()

    root.protocol( 'WM_DELETE_WINDOW', app.exit)
    try:
        app.start()
    except:
        app.exit()
