import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import numpy as np
import importlib
from Classes.Params import param
from pathlib import Path
cwd = os.path.split(os.path.abspath(__file__))[0]  
#cwd = str(Path(Path(cwd).parent.absolute()).parent.absolute())
print(cwd)

# File of result
result_file = cwd + '/theta_tilde_Hessian_results.npy'
result_file2 = cwd + '/sum_diag_H_result.npy'

# After collecting results or in real time while training
REAL_TIME = 1

# Real Hessian in Clients
HESSIAN = 1

if REAL_TIME:
    # If file of result exists, delete it
    if os.path.isfile(result_file):
        os.remove(result_file)
    if os.path.isfile(result_file2):
        os.remove(result_file2)

# Create figure for plotting
fig = plt.figure()

if HESSIAN and param.ALGORITHM_NAME == 'FEDADPPROX':
    ax_theta_tilde = fig.add_subplot(1, 3, 1)
    ax_sum_diag_H_approx = fig.add_subplot(1, 3, 2)
    ax_sum_diag_H = fig.add_subplot(1, 3, 3)

elif param.ALGORITHM_NAME == 'FEDADPPROX':
    ax_theta_tilde = fig.add_subplot(1, 2, 1)
    ax_sum_diag_H_approx = fig.add_subplot(1, 2, 2)

elif HESSIAN:
    ax_sum_diag_H = fig.add_subplot(1, 1, 1)

x_theta_tilde = []
y_theta_tilde = []

x_sum_diag_H_approx = []
y_sum_diag_H_approx = []

x_sum_diag_H = []
y_sum_diag_H = []

i = 0
ani = 0

# This function is called periodically from FuncAnimation
def animate(i):

    global x_theta_tilde, y_theta_tilde, x_sum_diag_H_approx, y_sum_diag_H_approx, x_sum_diag_H, y_sum_diag_H
    global ani
    importlib.reload(param)

    if REAL_TIME:
        # For first round
        try:

            if param.ALGORITHM_NAME == 'FEDADPPROX':
                result = np.load(result_file, allow_pickle = True)

            if HESSIAN:
                result2 = np.load(result_file2, allow_pickle = True)


            if param.ALGORITHM_NAME == 'FEDADPPROX':
                while len(result[-1]['theta_tilde']) < 20:
                    result[-1]['theta_tilde'].append(None)
                    result[-1]['sum_diag_H'].append(None)

            if HESSIAN:
                while len(result2[-1]['sum_diag_H']) < 20:
                    result2[-1]['sum_diag_H'].append(None)

            if param.ALGORITHM_NAME == 'FEDADPPROX':
                x_theta_tilde.append(result[-1]['round']) # or 'time_round'
                y_theta_tilde.append(result[-1]['theta_tilde'])
                x_sum_diag_H_approx.append(result[-1]['round']) # or 'time_round'
                y_sum_diag_H_approx.append(result[-1]['sum_diag_H'])

            if HESSIAN:
                x_sum_diag_H.append(result2[-1]['round']) # or 'time_round'
                y_sum_diag_H.append(result2[-1]['sum_diag_H'])
                                

            # print(x_theta_tilde, y_theta_tilde)
            # print(x_sum_diag_H_approx, y_sum_diag_H_approx)
            # print('\n')

            # x_theta_tilde = x_theta_tilde[-20:]
            # y_theta_tilde = y_theta_tilde[-20:]

            # If data are updated
            if param.ALGORITHM_NAME == 'FEDADPPROX':
                if x_sum_diag_H_approx[-1] != x_sum_diag_H_approx[-2]:
                    x_sum_diag_H_approx = x_sum_diag_H_approx[-20:]
                    y_sum_diag_H_approx = y_sum_diag_H_approx[-20:]
            # if HESSIAN:
            #     if x_sum_diag_H[-1] != x_sum_diag_H[-2]:
            #         x_sum_diag_H = x_sum_diag_H[-20:]
            #         y_sum_diag_H= y_sum_diag_H[-20:]    
                                    

            if param.ALGORITHM_NAME == 'FEDADPPROX':
                # Draw accuracy
                ax_theta_tilde.clear()
                ax_theta_tilde.plot(x_theta_tilde, y_theta_tilde)

                # Format plot
                plt.sca(ax_theta_tilde)   # Use the pyplot interface to change just one subplot
                plt.xticks(rotation=45, ha='right')
                plt.subplots_adjust(bottom=0.30)
                ax_theta_tilde.set(xlabel='Rounds', ylabel='Angle')
                ax_theta_tilde.title.set_text('Theta_tilde')
                ax_theta_tilde.grid()
                ax_theta_tilde.legend(['Client: {}'.format(client_) for client_ in range(param.NUM_CLIENTS)], loc='lower right', shadow=True)

                # Draw sum_diag_H_approx
                ax_sum_diag_H_approx.clear()
                ax_sum_diag_H_approx.plot(x_sum_diag_H_approx, y_sum_diag_H_approx)

                # Format plot
                plt.sca(ax_sum_diag_H_approx)  # Use the pyplot interface to change just one subplot
                plt.xticks(rotation=45, ha='right')
                plt.subplots_adjust(bottom=0.30)
                ax_sum_diag_H_approx.set(xlabel='Rounds', ylabel='sum_diag_H_approx')
                ax_sum_diag_H_approx.title.set_text('sum_diag_H_approx')
                ax_sum_diag_H_approx.grid()
                ax_sum_diag_H_approx.legend(['Client: {}'.format(client_) for client_ in range(param.NUM_CLIENTS)], loc='upper right', shadow=True)

            if HESSIAN:
                # Draw sum_diag_H
                ax_sum_diag_H.clear()
                ax_sum_diag_H.plot(x_sum_diag_H, y_sum_diag_H)

                # Format plot
                plt.sca(ax_sum_diag_H)  # Use the pyplot interface to change just one subplot
                plt.xticks(rotation=45, ha='right')
                plt.subplots_adjust(bottom=0.30)
                ax_sum_diag_H.set(xlabel='Rounds', ylabel='sum_diag_H')
                ax_sum_diag_H.title.set_text('sum_diag_H')
                ax_sum_diag_H.grid()
                ax_sum_diag_H.legend(['Client: {}'.format(client_) for client_ in range(param.NUM_CLIENTS)], loc='upper right', shadow=True)


            i+= 1

        except:

            pass
    else:

        if param.ALGORITHM_NAME == 'FEDADPPROX':
            result = np.load(result_file, allow_pickle = True)

        if HESSIAN:
            result2 = np.load(result_file2, allow_pickle = True)

        # print(result2[0].keys())

        if param.ALGORITHM_NAME == 'FEDADPPROX':
            while len(result[i]['theta_tilde']) < 20:
                result[i]['theta_tilde'].append(None)
                result[i]['sum_diag_H'].append(None)

        if HESSIAN:
            while len(result2[i]['sum_diag_H']) < 20:
                result2[i]['sum_diag_H'].append(None)



        if param.ALGORITHM_NAME == 'FEDADPPROX':
            x_theta_tilde.append(result[i]['round']) # or 'time_round'
            y_theta_tilde.append(result[i]['theta_tilde'])
            x_sum_diag_H_approx.append(result[i]['round']) # or 'time_round'
            y_sum_diag_H_approx.append(result[i]['sum_diag_H'])

        if HESSIAN:
            x_sum_diag_H.append(result2[i]['round']) # or 'time_round'
            y_sum_diag_H.append(result2[i]['sum_diag_H'])


        # print(x_theta_tilde, y_theta_tilde)
        # print(x_sum_diag_H_approx, y_sum_diag_H_approx)
        # print('\n')

        # x_theta_tilde = x_theta_tilde[-20:]
        # y_theta_tilde = y_theta_tilde[-20:]

        # If data are updated
        if param.ALGORITHM_NAME == 'FEDADPPROX':
            try:
                if x_sum_diag_H_approx[-1] != x_sum_diag_H_approx[-2]:
                    x_sum_diag_H_approx = x_sum_diag_H_approx[-20:]
                    y_sum_diag_H_approx = y_sum_diag_H_approx[-20:]
            except:
                pass

        # if HESSIAN:
        #     try:
        #         if x_sum_diag_H[-1] != x_sum_diag_H[-2]:
        #             x_sum_diag_H = x_sum_diag_H[-20:]
        #             y_sum_diag_H= y_sum_diag_H[-20:]     
        #     except:
        #         pass

        if param.ALGORITHM_NAME == 'FEDADPPROX':
            # Draw accuracy
            ax_theta_tilde.clear()
            ax_theta_tilde.plot(x_theta_tilde, y_theta_tilde)

            # Format plot
            plt.sca(ax_theta_tilde)   # Use the pyplot interface to change just one subplot
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            ax_theta_tilde.set(xlabel='Rounds', ylabel='Angle')
            ax_theta_tilde.title.set_text('Theta_tilde')
            ax_theta_tilde.grid()
            ax_theta_tilde.legend(['Client: {}'.format(client_) for client_ in range(param.NUM_CLIENTS)], loc='lower right', shadow=True)

            # Draw loss
            ax_sum_diag_H_approx.clear()
            ax_sum_diag_H_approx.plot(x_sum_diag_H_approx, y_sum_diag_H_approx)

            # Format plot
            plt.sca(ax_sum_diag_H_approx)  # Use the pyplot interface to change just one subplot
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            ax_sum_diag_H_approx.set(xlabel='Rounds', ylabel='sum_diag_H_approx')
            ax_sum_diag_H_approx.title.set_text('sum_diag_H_approx')
            ax_sum_diag_H_approx.grid()
            ax_sum_diag_H_approx.legend(['Client: {}'.format(client_) for client_ in range(param.NUM_CLIENTS)], loc='upper right', shadow=True)

        if HESSIAN:
            # Draw sum_diag_H
            ax_sum_diag_H.clear()
            ax_sum_diag_H.plot(x_sum_diag_H, y_sum_diag_H)

            # Format plot
            plt.sca(ax_sum_diag_H)  # Use the pyplot interface to change just one subplot
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            ax_sum_diag_H.set(xlabel='Rounds', ylabel='sum_diag_H')
            ax_sum_diag_H.title.set_text('sum_diag_H')
            ax_sum_diag_H.grid()
            ax_sum_diag_H.legend(['Client: {}'.format(client_) for client_ in range(param.NUM_CLIENTS)], loc='upper right', shadow=True)


        i+= 1 
        if i == 100:
            ani.event_source.stop()

if REAL_TIME:
    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, interval=2000)
    plt.show()

else:
    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, interval=10)
    plt.show()



# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax_theta_tilde = fig.add_subplot(1, 2, 1)
# x_theta_tilde = [0, 1, 2]
# y_theta_tilde = [[2,3,4],[3, 4, 5],[4]]
# while len(result[-1]['round'] < 20):
#     result[-1]['round'].append(None)
# ax_theta_tilde.plot(x_theta_tilde[:2], y_theta_tilde[:2])
# plt.show()
