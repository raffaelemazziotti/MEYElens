
from psychopy import visual, core, monitors
from psychopy.hardware import keyboard
from game_lib import *

mon = monitors.Monitor('TestMonitor')#fetch the most recent calib for this monitor
#mon.setGamma(1.5)

win = visual.Window(size=(800, 800), pos=None, color=(0, 0, 0), units='norm',monitor=mon)
kb = keyboard.Keyboard()
keys_list = ['left', 'right', 'down', 'escape', 'q' , 'w', 'e']

frame = Frame(win)
striker = Striker(win,border=1.9)
ball = Ball(win,contrast=1,speed=0.01)

bricks = list()
bricks.append( Brick(win,pos=(-.6,.6), size=(.4,.25)) )
bricks.append( Brick(win,pos=(-0,.6), size=(.4,.25)) )
bricks.append( Brick(win,pos=(.6,.6), size=(.4,.25)) )

win.flip()

while True: # main loop


    ball.update()
    if ball.collision(striker):
        ball.hit()

    for i,b in enumerate(bricks):
        if ball.collision(b):
            #ball.hit()
            b.hit()
            if b.is_dead:
                bricks.pop(i)

    frame.update()
    win.flip()
    keys = kb.getKeys(keys_list, waitRelease=False, clear=False)
    if keys:
        #print([k.name for k in keys])
        keys = keys[-1]
        if keys.name == 'escape':
            print('##### KEY escape: EXIT')
            break
        else:
            if keys.duration is None:
                if keys.name == 'left':
                    striker.moveLeft()
                if keys.name == 'right':
                    striker.moveRight()
                #print(striker.get_posx())
            else:
                kb.clearEvents()

# exit routine
win.close()
core.quit()

