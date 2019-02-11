# -*- coding: UTF-8 -*-
import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np


def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)

def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="file:/home/hduser/pythonsparkexample/PythonProject/"
    else:   
        Path="hdfs://master:9000/user/hduser/"
#如果您要在cluster模式執行(hadoop yarn 或Spark Stand alone)，請依照書上說明，先上傳檔案至HDFS目錄

def extract_label(field):
    label=(field[0])
    return float(label)

def extract_features(x):
    categoryFeatures = [float(a) for a in x[1:]]
    return categoryFeatures
    
def PrepareData(sc): 
    #----------------------1.匯入並轉換資料-------------
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile(Path+"data/DRtrain.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    lines = rawData.map(lambda x: x.split(","))
    print("共計：" + str(lines.count()) + "筆")
#----------2.建立訓練評估所需資料 RDD[LabeledPoint]-------    
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))
        
    #-----------3.以隨機方式將資料分為3部份並且回傳-------------
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("將資料分為trainData:" + str(trainData.count()) + 
              "   validationData:" + str(validationData.count()) +
              "   testData:" + str(testData.count()))
    return (trainData, validationData, testData) #回傳資料

def extract_features1(x):
    categoryFeatures = [float(a) for a in x[:]]
    return categoryFeatures

def plot_image(images, idx, num):
    image_data = images.take(num)
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:
        num = 25
    for i in range(0,num):
        target = image_data[idx]
        prediction = model.predict(target[1])
        ax = plt.subplot(5,5,1+i)
        ax.imshow(target[0], cmap='binary')
        title = "prediction:"+ str(prediction)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx += 1
    plt.show()

def PredictData(sc,model): 
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile(Path+"data/DRtest.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)
    line = rawData.map(lambda x: x.split(","))
    print("共計：" + str(line.count()) + "筆")
    dataRDD = line.map(lambda r:(np.array(extract_features1(r)).reshape((28,28)), extract_features1(r)))
    plot_image(dataRDD, 0, 10)
 


def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData.map(lambda p: p.label))
    metrics = MulticlassMetrics(scoreAndLabels)
    accuracy = metrics.accuracy
    return(accuracy)
 

def trainEvaluateModel(trainData,validationData,
                                          impurityParm, maxDepthParm, maxBinsParm):
    startTime = time()
    model=DecisionTree.trainClassifier( \
        trainData, numClasses=10, categoricalFeaturesInfo={}, \
        impurity="entropy", maxDepth=5, maxBins=5)
    accuracy = evaluateModel(model, validationData)
    duration = time() - startTime
    print    "訓練評估：使用參數" + \
                " impurityParm= %s"%impurityParm+ \
                " maxDepthParm= %s"%maxDepthParm+ \
                " maxBinsParm = %d."%maxBinsParm + \
                 " 所需時間=%d"%duration + \
                 " 結果accuracy = %f " % accuracy 
    return (accuracy,duration, impurityParm, maxDepthParm, maxBinsParm,model)




def evalParameter(trainData, validationData, evaparm,impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData,  impurity,numIter,  maxBins  ) 
               for impurity in impurityList for numIter in maxDepthList  for maxBins in maxBinsList ]
    if evaparm=="impurity":
        IndexList=impurityList[:]
    elif evaparm=="maxDepth":
        IndexList=maxDepthList[:]
    elif evaparm=="maxBins":
        IndexList=maxBinsList[:]
    df = pd.DataFrame(metrics,index=IndexList,
               columns=['accuracy', 'duration','impurity', 'maxDepth', 'maxBins','model'])
    
    showchart(df,evaparm,'accuracy','duration',0.6,1.0 )
    
    
def showchart(df,evalparm ,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar', titl =evalparm,figsize=(10,6),legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[lineData ]].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    plt.show()
    
    
def evalAllParameter(training_RDD, validation_RDD, impurityList, maxDepthList, maxBinsList):    
    metrics = [trainEvaluateModel(trainData, validationData,  impurity,numIter,  maxBins  ) 
                        for impurity in impurityList for numIter in maxDepthList  for maxBins in maxBinsList ]
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter=Smetrics[0]
    print("調校後最佳參數：impurity:" + str(bestParameter[2]) + 
             "  ,maxDepth:" + str(bestParameter[3]) + 
            "  ,maxBins:" + str(bestParameter[4])   + 
            "  ,結果accuracy = " + str(bestParameter[0]))
    return bestParameter[5]

def  parametersEval(training_RDD, validation_RDD):
    print("----- 評估impurity參數使用 ---------")
    evalParameter(trainData, validationData,"impurity", 
                              impurityList=["gini", "entropy"],   
                              maxDepthList=[10],  
                              maxBinsList=[10 ])  

    print("----- 評估maxDepth參數使用 ---------")
    evalParameter(trainData, validationData,"maxDepth", 
                              impurityList=["gini"],                    
                              maxDepthList=[3, 5, 10, 15, 20, 25],    
                              maxBinsList=[10])   

    print("----- 評估maxBins參數使用 ---------")
    evalParameter(trainData, validationData,"maxBins", 
                              impurityList=["gini"],      
                              maxDepthList =[10],        
                              maxBinsList=[3, 5, 10, 50, 100, 200 ])
    


def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("RunDecisionTreeBinary")                         \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    return (sc)

if __name__ == "__main__":
    print("RunDecisionTreeMulti")
    sc=CreateSparkContext()
    print("==========資料準備階段===============")
    (trainData, validationData, testData) =PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("==========訓練評估階段===============")
    (AUC,duration, impurityParm, maxDepthParm, maxBinsParm,model)= \
        trainEvaluateModel(trainData, validationData, "entropy", 15,50)
    if (len(sys.argv) == 2) and (sys.argv[1]=="-e"):
        parametersEval(trainData, validationData)
    elif   (len(sys.argv) == 2) and (sys.argv[1]=="-a"): 
        print("-----所有參數訓練評估找出最好的參數組合---------")  
        model=evalAllParameter(trainData, validationData,
                          ["gini", "entropy"],
                          [3, 5, 10, 15],
                          [3, 5, 10, 50 ])
                
    print("==========測試階段===============")
    accuracy = evaluateModel(model, testData)
    print("使用testata測試最佳模型,結果 accuracy:" + str(accuracy))
    print("==========預測資料===============")
    PredictData(sc, model)
    #print   model.toDebugString()
    