from psychopy import visual
import numpy as np
from shapely.geometry import Polygon
from numpy import random

class Frame:
    def __init__(self,win):
        self.frame = visual.rect.Rect(win, size=1.9, lineWidth=2, lineColor=(-1,-1,-1), fillColor=False, autoDraw=True, units='norm')
        self.frame.opacity = 0.7
        self.frame.draw()
        self.frame.autodraw=False


    def draw(self):
        self.frame.draw()

    def update(self):
        #self.frame.buildNoise()
        self.frame.draw()


    def autodraw(self,bool):
        self.frame.autodraw = bool

    def getVertices(self):
        return Polygon( self.frame.verticesPix )

class Striker:

    def __init__(self,win,size=(.4,.05), border=None):
        self.striker = visual.GratingStim(win,size=size, mask=None, units='norm', sf=4, ori=0, pos=(0,-.9), autoDraw=True)
        self.size = size
        self.vertices = self.striker.verticesPix
        self.id = 'striker'

        if border is None:
            self.border = (2 - self.size[0])/2
        else:
            self.border = (border - self.size[0])/2

    def draw(self):
        self.striker.draw()

    def autodraw(self,bool):
        self.striker.autodraw = bool

    def moveLeft(self):
        if self.striker.pos[0] > -self.border:
            self.striker.pos = (self.striker.pos[0]-0.01, self.striker.pos[1])

    def moveRight(self):
        if self.striker.pos[0] < self.border:
            self.striker.pos = (self.striker.pos[0]+0.01, self.striker.pos[1])

    def get_posx(self):
        return self.striker.pos[0]

    def getVertices(self):
        return Polygon( self.striker.verticesPix )

class Ball:

    def __init__(self,win, size=(.1,.1), border=None, contrast=0.8,speed=0.008 ):
        self.ball = visual.GratingStim(win, size=size, mask='gauss', units='norm', sf=3, ori=0, pos=(0, -.7),
                                             autoDraw=True, blendmode='avg')

        self.ball.opacity = contrast
        self.ball.contrast = contrast

        self.size = size
        self.speed = speed
        self.xFac, self.yFac = 1, 1
        self.poly = Polygon( self.ball.verticesPix )
        self.id = 'ball'

        if border is None:
            self.border = ((2 - self.size[0]) / 2, (2 - self.size[1]) / 2)
        else:
            self.border = ( (border[0] - self.size[0]) / 2, (border[1] - self.size[1]) / 2)

    def draw(self):
        self.ball.draw()

    def autodraw(self,bool):
        self.ball.autodraw = bool

    def get_pos(self):
        return self.ball.pos

    def getVertices(self):
        self.poly = Polygon( ( (self.ball.verticesPix[0],self.ball.verticesPix[1]), ) )
        return self.poly

    def update(self):
        self.ball.pos = (self.ball.pos[0] + self.xFac * self.speed, self.ball.pos[1] + self.yFac * self.speed )
        if np.abs(self.ball.pos[0]) > self.border[0] - (self.size[0]/2):
            self.xFac *= -1
        if np.abs(self.ball.pos[1]) > self.border[1] - (self.size[1]/2):
            self.yFac *= -1
        self.poly = Polygon(self.ball.verticesPix)

    def hit(self):
        pass
        #self.yFac *= -1

    def collision(self, col_obj):
        rect = col_obj.getVertices()
        if self.poly.intersects( rect ):
            self.yFac *= -1
            if col_obj.id is 'striker':
                if self.poly.centroid.x > rect.centroid.x:
                    self.xFac = 1
                elif self.poly.centroid.x < rect.centroid.x:
                    self.xFac = -1
            return True
        else:
            return False

class Brick:

    def __init__(self,win, size=(.4,.3), pos=(0,0), health=2 ):

        self.brick = visual.GratingStim(win, size=size, mask=None, units='norm', sf=3, ori=0, pos=pos,
                                             autoDraw=True)
        self.health = health
        self.is_dead = False
        self.id = 'brick'

    def draw(self):
        self.brick.draw()

    def autodraw(self,bool):
        self.brick.autodraw = bool

    def hit(self):
        self.health -=1
        if self.health <= 0:
            self.brick.opacity = 0
            self.is_dead = True

    def getVertices(self):
        return Polygon( self.brick.verticesPix )

    def id(self):
        return 'brick'