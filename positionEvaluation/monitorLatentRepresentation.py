import matplotlib.pyplot
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = matplotlib.pyplot.figure()
ax1 = fig.add_subplot(1, 1, 1)

predictedTargetClassToColorDict = {(0, 0): 'red', (1, 0): 'salmon', (1, 1): 'blue', (0, 1): 'lightblue'}
targetClassToMarkerStyle = {0: 'o', 1: '^'}

def animate(i):
    """graph_data = open('./outputs/epochLoss.csv', 'r').read()
    lines = graph_data.split('\n')
    #print ("animate(): lines = {}".format(lines))
    epochs = []
    trainingLossList = []
    validationLossList = []
    averageRewardList = []
    winRateList = []
    drawRateList = []
    lossRateList = []

    headers = lines[0].split(',')
    for lineNdx in range (1, len(lines)):
        line = lines[lineNdx]
        if len(line) > 1 :
            epoch, trainingLoss, validationLoss, averageReward, winRate, drawRate, lossRate = \
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
            averageRewardList.append(float(averageReward))
            winRateList.append(float(winRate))
            drawRateList.append(float(drawRate))
            lossRateList.append(float(lossRate))

    ax1.clear()
    plt.ylim((0, 1.0))
    ax1.plot(epochs, trainingLossList, label=headers[1])
    ax1.plot(epochs, validationLossList, label=headers[2])
    ax1.plot(epochs, averageRewardList, label=headers[3])
    ax1.plot(epochs, winRateList, label=headers[4])
    ax1.plot(epochs, drawRateList, label=headers[5])
    ax1.plot(epochs, lossRateList, label=headers[6])

    ax1.legend(shadow=True, fancybox=True)
    """
    graph_data = open('./outputs/latentRepresentation.csv', 'r').read()
    lines = graph_data.split('\n')
    #print ("lines = {}".format(lines))
    if len(lines) > 1:
        lastLine = lines[-2]
        #print ("lastLine = {}".format(lastLine))
        values = lastLine.split(',')
        #print ("values = {}".format(values))
        if len(values) % 4 != 0:
            raise ValueError("The number of comma-separated values ({}) is not a multiple of 4".format(len(values)))
        numberOfPoints = len(values) // 4
        x0List = []
        x1List = []
        predictedClassList = []
        targetClassList = []
        colorList = []
        #markerList = []
        for pointNdx in range(numberOfPoints):
            x0 = float(values[4 * pointNdx])
            x1 = float(values[4 * pointNdx + 1])
            x0List.append(x0)
            x1List.append(x1)
            predictedClass = int(float(values[4 * pointNdx + 2]))
            predictedClassList.append(predictedClass)
            targetClass = int(float(values[4 * pointNdx + 3]))
            targetClassList.append(targetClass)
            color = predictedTargetClassToColorDict[(predictedClass, targetClass)]
            colorList.append(color)
            #markerList.append(targetClassToMarkerStyle[targetClass])
        ax1.clear()
        matplotlib.pyplot.scatter(x0List, x1List, c=colorList, marker='o')



ani = animation.FuncAnimation(fig, animate, interval=1000)
matplotlib.pyplot.show()