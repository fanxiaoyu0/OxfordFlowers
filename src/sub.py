import matplotlib.pyplot as plt

# classA_grades = [80,85,90,95,100]
# classB_grades = [30,60,40,50,80]
# grades_range = [0,25, 50,75,100]

# plt.plot( grades_range, classA_grades, 'ro')
# plt.plot( grades_range, classB_grades, 'go')
# epoch: 17 validAccuracy: 0.2196 trainAccuracy: 0.2608 delta: -0.0039
def draw_accuracy_curve():
    epochList=[i for i in range(279)]
    print(epochList)
    trainAccuracyList=[]
    validAccuracyList=[]
    with open("ConvMixerBest.txt") as f:
        lines=f.readlines()
        for line in lines:
            wordList=line.split(" ")
            validAccuracyList.append(float(wordList[3]))
            trainAccuracyList.append(float(wordList[5]))
    print(trainAccuracyList)
    print(validAccuracyList)
    plt.plot( epochList, trainAccuracyList, 'r')
    plt.plot( epochList, validAccuracyList, 'g')
    plt.savefig("ConvMixerBest.png")

if __name__=="__main__":
    draw_accuracy_curve()
    print("All is well!")