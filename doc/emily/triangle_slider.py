'''
https://en.wikipedia.org/wiki/Triangular_distribution
Triangular distribution:
    * lower limit a,
    * upper limit b and
    * mode c
where a < b and a ≤ c ≤ b. 
'''

import numpy as np
from scipy.stats import triang
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# The parametrized function to be plotted
def f(t, c):
    return triang.pdf(t, c)  # triangular pdf

t = np.linspace(0, 1, 1000)   # sample at 1000 points for plotting
init_c = .5  # initial parameter

fig, ax = plt.subplots()
line, = ax.plot(t, f(t, init_c))  # line plot to manipulate
ax.set_xlabel('x')

fig.subplots_adjust(left=0.25,
                    bottom=0.25)  # adjust plot to make room for sliders

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])  # horizontal slider to control C
c_slider = Slider(ax=axfreq,
                  label='C (mode)',
                  valmin=0.,
                  valmax=1.,
                  valinit=init_c)

def update(val):  # function to be called when sliders value changes
    line.set_ydata(f(t, c_slider.val))
    fig.canvas.draw_idle()

c_slider.on_changed(update)  # register update function with slider
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])  # button to reset to initial value
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    c_slider.reset()
button.on_clicked(reset)

plt.show()  # show the plot!
