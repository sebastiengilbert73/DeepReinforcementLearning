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
    averageActionValuesTrainingLossList = []
    averageRewardAgainstRandomPlayerList = []
    winRateList = []
    drawRateList = []
    lossRateList = []
    headers = lines[0].split(',')
    for lineNdx in range (1, len(lines)):
        line = lines[lineNdx]
        if len(line) > 1 :
            epoch, averageActionValuesTrainingLoss, averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate = \
             line.split(',')
            epochs.append(int(epoch))
            try:
                averageActionValuesTrainingLossList.append(float(averageActionValuesTrainingLoss))
            except ValueError:
                averageActionValuesTrainingLossList.append(None)
            averageRewardAgainstRandomPlayerList.append(float(averageRewardAgainstRandomPlayer))
            winRateList.append(float(winRate))
            drawRateList.append(float(drawRate))
            lossRateList.append(float(lossRate))
    ax1.clear()
    ax1.plot(epochs, averageActionValuesTrainingLossList, label=headers[1])
    ax1.plot(epochs, averageRewardAgainstRandomPlayerList, label=headers[2])
    ax1.plot(epochs, winRateList, label=headers[3])
    ax1.plot(epochs, drawRateList, label=headers[4])
    ax1.plot(epochs, lossRateList, label=headers[5])
    ax1.legend(shadow=True, fancybox=True)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()