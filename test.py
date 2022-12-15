import datetime
import scipy
import numpy as np
import sklearn
import torch

def parse_expenses(expenses_string):
    """Parse the list of expenses and return the list of triples (date, value, currency).
    Ignore lines starting with #.
    Parse the date using datetime.
    Example expenses_string:
        2016-01-02 -34.01 USD
        2016-01-03 2.59 DKK
        2016-01-03 -2.72 EUR
    """
    expenses = []
    for line in expenses_string.splitlines():
        if line.startswith('#'):
            continue
        date, value, currency = line.split()
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        value = float(value)
        expenses.append((date, value, currency))
    return expenses

#create a sine wave in the range of 0 to 2pi and print it in a graph
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()

#create a wave animation using matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([], [], lw=2)
ax.set_ylim(-1, 1)
ax.set_xlim(0, 2*np.pi)
ax.grid()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10.0))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), interval=25, blit=False)
plt.show()


