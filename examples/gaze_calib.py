from meyelens.gaze import ScreenPositions, GazeData, GazeModelPoly
from psychopy import visual, core
from psychopy.hardware import keyboard
from meyelens.online import MeyeRecorder
from meyelens.utils import CountdownTimer
import numpy as np

# Initialize variables
dot_fixation = 2
wait_time = 1
calib_full = False
screen_size = (1000,1000)
dot_radius = 12


# MeyeRecorder setup
meye_reco = MeyeRecorder(show_preview=False, filename='track_cal')#,cam_crop=[90,210,260,280])
meye_reco.preview()

# Psychopy setup
kb = keyboard.Keyboard()
win = visual.Window(size=screen_size, units="pix", color=(-1, -1, -1), fullscr=False) # useFBO=True
circle = visual.Circle(win=win, radius=10, fillColor=(1,1,1))

meye_reco.start()

stims = ScreenPositions(screen_size[0]*0.9,screen_size[1]*0.9)


# timers
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
win.flip()

# Train model
gdata = GazeData()
gaze_points,screen_positions = gdata.get_last()
gmodel = GazeModelPoly()
gmodel.train(gaze_points,screen_positions)

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
core.quit()