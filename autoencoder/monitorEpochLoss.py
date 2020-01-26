import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    graph_data = open('./outputs/epochLoss.csv', 'r').read()
    lines = graph_data.split('\n')
    #print ("animate(): lines = {}".format(lines))
    epochs = []
    trainingLossList = []
    validationLossList = []
    errorRateList = []

    headers = lines[0].split(',')
    for lineNdx in range (1, len(lines)):
        line = lines[lineNdx]
        if len(line) > 1 :
            epoch, trainingLoss, validationLoss, errorRate = \
             line.split(',')
            epochs.append(int(epoch))
            try:
                trainingLossList.append(float(trainingLoss))
            except ValueError:
                trainingLossList.append(None)
            validationLossList.append(float(validationLoss))
            errorRateList.append(float(errorRate))

    ax1.clear()
    plt.ylim((0, 1.0))
    ax1.plot(epochs, trainingLossList, label=headers[1])
    ax1.plot(epochs, validationLossList, label=headers[2])
    ax1.plot(epochs, errorRateList, label=headers[3])

    ax1.legend(shadow=True, fancybox=True)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()