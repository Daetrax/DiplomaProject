import os, cv2
import matplotlib.pyplot as plt

def visualizeMasks(directory):
    plotdata = {"White": [] , "Black": [], "W_B_Ratio": []}
    nonEmptyCount = 0
    for name in os.listdir(directory):
        if("mask" in name):
            continue


        filename = directory + name
        img = cv2.imread(filename)
        mask = cv2.imread(filename.replace(".tif", "") + "_mask.tif", cv2.IMREAD_GRAYSCALE)

        if cv2.countNonZero(mask) != 0:
            nonEmptyCount += 1
        white = cv2.countNonZero(mask)
        black = (mask.shape[0] * mask.shape[1]) - white

        plotdata["White"].append(white)
        plotdata["Black"].append(black)
        plotdata["W_B_Ratio"].append(white / (black + white))
        print(white / (black + white))

    globalAvg = 0
    globalAvgNoEmptyMask = 0
    for avg in plotdata["W_B_Ratio"]:
        globalAvg += avg
        globalAvgNoEmptyMask += avg

    globalAvg /= len(plotdata["W_B_Ratio"])
    globalAvgNoEmptyMask /= nonEmptyCount

    print("Global average: ", globalAvg * 100)
    print("Global average without empty masks: ", globalAvgNoEmptyMask * 100)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.bar(plotdata["White"], plotdata["Black"])
    # plt.subplot(221)
    # plt.plot(plotdata["White"], plotdata["Black"], 'b--')
    plt.xlabel('White')
    plt.ylabel('Black')
    plt.title('Ratio of black and white pixels in mask')

    plt.show()

