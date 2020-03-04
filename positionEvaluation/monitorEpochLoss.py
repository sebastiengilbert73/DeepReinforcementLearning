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
    validationAccuracyList = []
    averageRewardList = []
    winRateList = []
    drawRateList = []
    lossRateList = []

    headers = lines[0].split(',')
    for lineNdx in range (1, len(lines)):
        line = lines[lineNdx]
        if len(line) > 1 :
            epoch, trainingLoss, validationLoss, validationAccuracy, averageReward, winRate, drawRate, lossRate = \
             line.split(',')
            epochs.append(int(epoch))
            try:
                trainingLossList.append(float(trainingLoss))
            except ValueError:
                trainingLossList.append(None)
            try:
                validationLossList.append(float(validationLoss))
            except ValueError:
                validationLossList.append(None)
            try:
                validationAccuracyList.append(float(validationAccuracy))
            except:
                validationAccuracyList.append(None)

            averageRewardList.append(float(averageReward))
            winRateList.append(float(winRate))
            drawRateList.append(float(drawRate))
            lossRateList.append(float(lossRate))

    ax1.clear()
    plt.ylim((0, 1.0))
    ax1.plot(epochs, trainingLossList, label=headers[1])
    ax1.plot(epochs, validationLossList, label=headers[2])
    ax1.plot(epochs, validationAccuracyList, label=headers[3])
    ax1.plot(epochs, averageRewardList, label=headers[4])
    ax1.plot(epochs, winRateList, label=headers[5])
    ax1.plot(epochs, drawRateList, label=headers[6])
    ax1.plot(epochs, lossRateList, label=headers[7], c='fuchsia')

    ax1.legend(shadow=True, fancybox=True, loc='upper left')

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()