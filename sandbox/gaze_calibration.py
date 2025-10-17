from meyelens.meye import MeyeAsyncRecorder
from visual_lib import CountdownTimer
from psychopy import visual, core, monitors
from psychopy.hardware import keyboard
import ctypes
import numpy as np
from gaze import GazeData,GazeModelPoly, ScreenPositions

# gaze calibration matrix

# Initialize variables
dot_fixation = 2
wait_time = 1
calib_full = False

stims = ScreenPositions(90,80,7)

# Screen and monitor setup
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
mon = monitors.Monitor('default')
mon.setDistance(15)
mon.setSizePix(screensize)
mon.setWidth(53.3)

# MeyeRecorder setup
meye_reco = MeyeAsyncRecorder(show_preview=False, filename='track_cal')#,cam_crop=[90,210,260,280])
meye_reco.preview()

# Psychopy setup
kb = keyboard.Keyboard()
win = visual.Window(size=(1000, 1000), units="deg", color=(-1, -1, -1), monitor=mon, useFBO=True, fullscr=False)
circle = visual.Circle(win=win, radius=2, fillColor=(1,1,1))


meye_reco.start()
stim_timer = CountdownTimer(dot_fixation+wait_time)
wait_timer = CountdownTimer(wait_time)

is_calibrating=True
user_abort = False
while is_calibrating:

    if stim_timer.is_finished():
        pos = stims.next()
        if pos is None:
            meye_reco.stop()
            break
        circle.pos = pos
        circle.color = "gray"
        wait_timer.start()
        stim_timer.start()

    circle.draw()
    win.flip()

    # Gaze data collection
    if wait_timer.is_finished():
        circle.color = 'white'
        meye_reco.save_frame('nan',pos[0],pos[1])

    keys = kb.getKeys(waitRelease=True)
    if 'escape' in keys:
        user_abort = True
        meye_reco.stop()
        win.close()
        core.quit()

# Train model
gdata = GazeData()
gaze_points,screen_positions = gdata.get_last()
gmodel = GazeModelPoly()
gmodel.train(gaze_points,screen_positions)

# test model
while not(user_abort):
    data = meye_reco.get_data()
    if not any(np.isnan(data['centroid'])):
        predicted = gmodel.predict(np.array(data['centroid']).reshape(1, -1) )
        circle.pos = tuple(predicted[0])

    circle.draw()
    win.flip()

    keys = kb.getKeys(waitRelease=True)
    if 'escape' in keys:
        user_abort = True
        meye_reco.stop()
        win.close()
        core.quit()


meye_reco.close_all()
win.close()
core.quit()

