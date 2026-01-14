import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Discrete "dates" (could be filenames)
dates = ["2025-10-01", "2025-10-05", "2025-10-09", "2025-10-13"]

# ----------------------------
# Figure with 3 plots
# ----------------------------
fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(bottom=0.25, wspace=0.25)

# Demo baseline + placeholders (replace with your own raster arrays)
baseline = np.random.rand(80, 80)  # <-- this is your fixed "start date" raster

imL = axL.imshow(baseline)
axL.set_title("Left (baseline / fixed)")

imM = axM.imshow(np.random.rand(80, 80))
titleM = axM.set_title(f"Middle (date={dates[0]})")

imR = axR.imshow(np.random.rand(80, 80))
titleR = axR.set_title("Right (derived)")

# ----------------------------
# Slider
# ----------------------------
ax_sl = fig.add_axes([0.12, 0.12, 0.60, 0.04])
sl = Slider(ax_sl, "idx", 0, len(dates) - 1, valinit=0, valstep=1)

def update(i):
    i = int(i)
    date = dates[i]

    # ----------------------------
    # TODO: YOUR LOGIC GOES HERE
    # ----------------------------
    # Example idea:
    #   current = read_raster_for(date)        # (H,W) or (H,W,C)
    #   derived = your_burn_mapping(baseline, current)
    #
    # Then update plots:
    #   imM.set_data(current_display)
    #   imR.set_data(derived_display)
    #
    # Demo placeholders:
    current = np.random.rand(80, 80)
    derived = current - baseline

    imM.set_data(current)
    imR.set_data(derived)

    titleM.set_text(f"Middle (date={date})")
    titleR.set_text("Right (derived)")
    fig.canvas.draw_idle()

sl.on_changed(update)

# ----------------------------
# Autoplay + Repeat
# ----------------------------
playing = {"on": False}
repeat = {"on": True}

timer = fig.canvas.new_timer(interval=500)

def step():
    i = int(sl.val)
    if i >= len(dates) - 1:
        if repeat["on"]:
            sl.set_val(0)  # loop
        else:
            playing["on"] = False
            btn_play.label.set_text("Play")
            timer.stop()
    else:
        sl.set_val(i + 1)  # advance

timer.add_callback(step)

# Play/Pause button
ax_btn = fig.add_axes([0.75, 0.11, 0.20, 0.06])
btn_play = Button(ax_btn, "Play")

def on_play(_):
    playing["on"] = not playing["on"]
    btn_play.label.set_text("Pause" if playing["on"] else "Play")
    if playing["on"]:
        timer.start()
    else:
        timer.stop()

btn_play.on_clicked(on_play)

# Repeat toggle button
ax_rep = fig.add_axes([0.75, 0.03, 0.20, 0.06])
btn_rep = Button(ax_rep, "Repeat: ON")

def on_repeat(_):
    repeat["on"] = not repeat["on"]
    btn_rep.label.set_text("Repeat: ON" if repeat["on"] else "Repeat: OFF")

btn_rep.on_clicked(on_repeat)

plt.show()

    


    