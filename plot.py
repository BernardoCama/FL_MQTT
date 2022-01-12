# Plot losses and metrics of training

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Classes.Params import param
try:
    param.tf.config.set_visible_devices([], 'GPU')
    visible_devices = param.tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass  

from pathlib import Path
cwd = os.path.split(os.path.abspath(__file__))[0]  
#cwd = str(Path(Path(cwd).parent.absolute()).parent.absolute())
print(cwd)

# File of result
result_file = cwd + '/result.npy'

# After collecting results or in real time while training
REAL_TIME = 1
TIME_ROUNDS = 0
DELETE = 1

if DELETE:
    # If file of result exists, delete it
    if os.path.isfile(result_file):
        os.remove(result_file)

# Create figure for plotting
fig = plt.figure()
ax_acc = fig.add_subplot(1, 2, 1)
ax_loss = fig.add_subplot(1, 2, 2)

x_acc = []
y_acc = []

x_loss = []
y_loss = []

time_ = []

i = 0

# This function is called periodically from FuncAnimation
def animate(j):

    global x_acc, y_acc, x_loss, y_loss, i

    # For first round
    try:

        result = np.load(result_file, allow_pickle = True)
        #print(result)

        if REAL_TIME:
            x_acc.append(result[-1]['round']) # or 'time_round' 'round'
            y_acc.append(result[-1]['mean_accuracy'][0])
            x_loss.append(result[-1]['round']) # or 'time_round' 'round'
            y_loss.append(result[-1]['mean_loss'])

            if 'mean_loss_train' in result[-1].keys():
                y_acc[-1] = np.append(param.copy.copy(y_acc[-1]), param.copy.copy(result[-1]['mean_accuracy_train'][0]))
                y_loss[-1] = np.append(param.copy.copy(y_loss[-1]), param.copy.copy(result[-1]['mean_loss_train']))

        else:
            x_acc.append(result[i]['round']) # or 'time_round' 'round'
            y_acc.append(result[i]['mean_accuracy'][0])
            x_loss.append(result[i]['round']) # or 'time_round' 'round'
            y_loss.append(result[i]['mean_loss'])

            if 'mean_loss_train' in result[i].keys():
                y_acc[i] = np.append(param.copy.copy(y_acc[i]), param.copy.copy(result[i]['mean_accuracy_train'][0]))
                y_loss[i] = np.append(param.copy.copy(y_loss[i]), param.copy.copy(result[i]['mean_loss_train']))
            
            if 'time_round' in result[i].keys() and TIME_ROUNDS:
                time_.append(result[i]['time_round'])

        # print(i)
        # print(time_[i])
        if len(time_) != 0:
            if len(time_) == 1:
                x_acc[i] = x_acc[i]*time_[i]
                x_loss[i] = x_loss[i]*time_[i]
            else:
                if time_[i] > 3*time_[i-1]:
                   time_[i] = time_[i-1] 
                x_acc[i] = x_acc[i-1] + time_[i]
                x_loss[i] = x_loss[i-1] + time_[i]

        # print(x_acc, y_acc)
        # print(x_loss, y_loss)
        # print('\n')

        # x_acc = x_acc[-20:]
        # y_acc = y_acc[-20:]

        # If data are updated
        # if x_loss[-1] != x_loss[-2]:
        #     x_loss = x_loss[-40:]
        #     y_loss = y_loss[-40:]

        # Draw accuracy
        ax_acc.clear()
        ax_acc.plot(x_acc, y_acc)

        # Format plot
        plt.sca(ax_acc)   # Use the pyplot interface to change just one subplot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        # plt.ylim(0.7,1)
        ax_acc.set(xlabel='Rounds' if not TIME_ROUNDS else 'Seconds', ylabel='Accuracy')
        ax_acc.title.set_text('Accuracy')
        ax_acc.grid()
        if not 'mean_loss_train' in result[-1].keys():
            ax_acc.legend([metric.name if hasattr(metric, 'name') else metric.__name__ for metric in param.METRICS], loc='lower right', shadow=True)
        else:
            ax_acc.legend([metric.name + '_valid' if hasattr(metric, 'name') else metric.__name__ + '_valid' for metric in param.METRICS] + [metric.name + '_train' if hasattr(metric, 'name') else metric.__name__ + '_train' for metric in param.METRICS] , loc='lower right', shadow=True)

        # Draw loss
        ax_loss.clear()
        ax_loss.plot(x_loss, y_loss)

        # Format plot
        plt.sca(ax_loss)  # Use the pyplot interface to change just one subplot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        ax_loss.set(xlabel='Rounds' if not TIME_ROUNDS else 'Seconds', ylabel='Loss')
        ax_loss.title.set_text('Loss')
        ax_loss.grid()
        if not 'mean_loss_train' in result[-1].keys():
            ax_loss.legend([param.LOSS.name if hasattr(param.LOSS, 'name') else param.LOSS.__name__ ], loc='upper right', shadow=True)
        else:
            ax_loss.legend([param.LOSS.name + '_valid' if hasattr(param.LOSS, 'name') else param.LOSS.__name__ + '_valid' ] + [param.LOSS.name + '_train' if hasattr(param.LOSS, 'name') else param.LOSS.__name__ + '_train' ], loc='upper right', shadow=True)

        i+= 1

    except:

        pass

if REAL_TIME:
    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, interval=2000)
    plt.show()

else:
    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, interval=10)
    plt.show()

