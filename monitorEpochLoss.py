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
    for line in lines:
        if len(line) > 1 and not line.startswith('epoch') and not line.startswith('0'):
            epoch, averageActionValuesTrainingLoss, averageRewardAgainstRandomPlayer, winRate, drawRate, lossRate = \
             line.split(',')
            epochs.append(int(epoch))
            averageActionValuesTrainingLossList.append(float(averageActionValuesTrainingLoss))
            averageRewardAgainstRandomPlayerList.append(float(averageRewardAgainstRandomPlayer))
            winRateList.append(float(winRate))
            drawRateList.append(float(drawRate))
            lossRateList.append(float(lossRate))
    ax1.clear()
    ax1.plot(epochs, averageActionValuesTrainingLossList, label='averageActionValuesTrainingLoss')
    ax1.plot(epochs, averageRewardAgainstRandomPlayerList, label='averageRewardAgainstRandomPlayer')
    ax1.plot(epochs, winRateList, label='winRate')
    ax1.plot(epochs, drawRateList, label='drawRate')
    ax1.plot(epochs, lossRateList, label='lossRate')
    ax1.legend(shadow=True, fancybox=True)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()