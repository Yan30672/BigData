# -*- coding: UTF-8 -*-
import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler


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

def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

def genderIndex(x):
    if x=="male":
        return 1
    else:
        return 0
#將Embarked轉換為Idx數值
def embarkedIdx(x):
    if x[-1]=="C":
        return 1
    elif x[-1]=="Q":
        return 2
    else:
        return 3
    
def convert_float(x):
    if x =="":
        return 0
    else:
        return float(x)

#收集特徵
def extract_features(x):
    categoryFeatures = np.zeros(4)
    categoryFeatures[0] = genderIndex(x[3])
    if len(x[-1]) != 0:
        categoryFeatures[embarkedIdx(x)] = 1
    else:
        pass
    numericalFeatures=[x[2], convert_float(x[4]), x[5], x[6], convert_float(x[8])]
    return np.concatenate((categoryFeatures, numericalFeatures))

def extract_features1(x):
    categoryFeatures = np.zeros(4)
    categoryFeatures[0] = genderIndex(x[2])
    if len(x[-1]) != 0:
        categoryFeatures[embarkedIdx(x)] = 1
    else:
        pass
    numericalFeatures=[x[1], convert_float(x[3]), x[4], x[5], convert_float(x[7])]
    return np.concatenate((categoryFeatures, numericalFeatures))
    
def extract_label(x):
    label = x[1]
    return float(label)

def PrepareData(sc): 
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile(Path+"data/train.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)
    lines = rawData.map(lambda x: x.split(","))
    newline = lines.map(lambda x: x[:3]+x[5:])
    notnull = newline.map(lambda x: len(x[4])!=0).sum()
    noNull = newline.filter(lambda x: len(x[4])!=0)
    AgeSum = noNull.map(lambda x: x[4]).collect()
    x = 0
    for a in AgeSum:
        x += float(a)
    AgeAver = x/notnull
    newlines = newline.map(lambda x:  x[:4]+[str(AgeAver)]+x[5:] if len(x[4])==0 else x)
    print("共計：" + str(newlines.count()) + "筆") 
    #----------2.建立訓練評估所需資料 RDD[LabeledPoint]-------    
    labelRDD = newlines.map(lambda r: extract_label(r))
    featureRDD = newlines.map(lambda r: extract_features(r))
    for i in featureRDD.first():
        print(str(i)+",")
    print "標準化之後"
    stdScaler = StandardScaler(withMean = True, withStd=True).fit(featureRDD)
    ScalerFeatureRDD=stdScaler.transform(featureRDD)
    for i in ScalerFeatureRDD.first():
        print(str(i)+",")
    labelpoint = labelRDD.zip(ScalerFeatureRDD)
    labelpointRDD = labelpoint.map(lambda r:LabeledPoint(r[0],r[1]))    
    #-----------3.以隨機方式將資料分為3部份並且回傳-------------
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("將資料分為trainData:" + str(trainData.count()) + 
              "   validationData:" + str(validationData.count()) +
              "   testData:" + str(testData.count()))
    return (trainData, validationData, testData) #回傳資料

def PredictData(sc,model): 
    print("開始匯入資料...")
    rawDataWithHeader = sc.textFile(Path+"data/test.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)
    lines = rawData.map(lambda x: x.split(","))
    newline = lines.map(lambda x: x[:2]+x[4:10]+[x[-1]])
    notnull = newline.map(lambda x: len(x[3])!=0).sum()
    noNull = newline.filter(lambda x: len(x[3])!=0)
    AgeSum = noNull.map(lambda x: x[3]).collect()
    x = 0
    for a in AgeSum:
        x += float(a)
    AgeAver = x/notnull
    newlines = newline.map(lambda x:  x[:3]+[str(AgeAver)]+x[4:] if len(x[3])==0 else x)
    print("共計：" + str(newlines.count()) + "筆")
    dataRDD = newlines.map(lambda r:(r[0], extract_features1(r)))
    DescDict = {
           0: "死亡",
           1: "倖存"
     }
    for data in dataRDD.take(10):
        predictResult = model.predict(data[1])
        print " 名單編號：  " +str(data[0])+"\n" +\
                  "             ==>預測:"+ str(predictResult)+ \
                  " 說明:"+DescDict[predictResult] +"\n"
        
    
#評估模型的準確率
def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData \
                                   .map(lambda p: p.label))  \
                                   .map(lambda (x,y): (float(x),float(y)) )
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC=metrics.areaUnderROC
    return( AUC)

#訓練與評估的功能，並且計算訓練評估的時間
def trainEvaluateModel(trainData,validationData,
                                        numIterations, stepSize, regParam):
    startTime = time()
    model = SVMWithSGD.train(trainData, numIterations, stepSize, regParam)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print    "訓練評估：使用參數" + \
                " numIterations="+str(numIterations) +\
                " stepSize="+str(stepSize) + \
                " regParam="+str(regParam) +\
                 " 所需時間="+str(duration) + \
                 " 結果AUC = " + str(AUC) 
    return (AUC,duration, numIterations, stepSize, regParam,model)

#評估不同參數對於模型準確率的影響

def evalParameter(trainData, validationData, evalparm,
                  numIterationsList, stepSizeList, regParamList):
    
    metrics = [trainEvaluateModel(trainData, validationData,  
                                numIterations,stepSize,  regParam  ) 
                       for numIterations in numIterationsList
                       for stepSize in stepSizeList  
                       for regParam in regParamList ]
    
    if evalparm=="numIterations":
        IndexList=numIterationsList[:]
    elif evalparm=="stepSize":
        IndexList=stepSizeList[:]
    elif evalparm=="regParam":
        IndexList=regParamList[:]
    
    df = pd.DataFrame(metrics,index=IndexList,
            columns=['AUC', 'duration','numIterations', 'stepSize', 'regParam','model'])
    showchart(df,evalparm,'AUC','duration',0.5,0.7 )
    
def showchart(df,evalparm ,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar', title =evalparm,figsize=(10,6),legend=True, fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[[lineData ]].values, linestyle='-', marker='o', linewidth=2.0,color='r')
    plt.show()

#找出最好的參數組合    
 
def evalAllParameter(trainData, validationData, 
                     numIterationsList, stepSizeList, regParamList):    
    metrics = [trainEvaluateModel(trainData, validationData,  
                            numIterations,stepSize,  regParam  ) 
                      for numIterations in numIterationsList 
                      for stepSize in stepSizeList  
                      for  regParam in regParamList ]
    
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter=Smetrics[0]
    
    print("調校後最佳參數：numIterations:" + str(bestParameter[2]) + 
                                      "  ,stepSize:" + str(bestParameter[3]) + 
                                     "  ,regParam:" + str(bestParameter[4])   + 
                                      "  ,結果AUC = " + str(bestParameter[0]))
    
    return bestParameter[5]

#執行參數評估，並且以圖表呈現
def  parametersEval(trainData, validationData):
    print("----- 評估numIterations參數使用 ---------")
    evalParameter(trainData, validationData,"numIterations", 
                              numIterationsList= [1, 3, 5, 15, 25],   
                              stepSizeList=[100],  
                              regParamList=[1 ])  
    print("----- 評估stepSize參數使用 ---------")
    evalParameter(trainData, validationData,"stepSize", 
                              numIterationsList=[25],                    
                              stepSizeList= [10, 50, 100, 200],    
                              regParamList=[1])   
    print("----- 評估regParam參數使用 ---------")
    evalParameter(trainData, validationData,"regParam", 
                              numIterationsList=[25],      
                              stepSizeList =[100],        
                              regParamList=[0.01, 0.1, 1 ])


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
    print("RunSVMWithSGDBinary")
    sc=CreateSparkContext()
    print("==========資料準備階段===============")
    (trainData, validationData, testData) =PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    print("==========訓練評估階段===============")
    (AUC,duration, numIterations, stepSize, regParam,model)= \
          trainEvaluateModel(trainData, validationData, 3, 50, 1)
    if (len(sys.argv) == 2) and (sys.argv[1]=="-e"):
        parametersEval(trainData, validationData)
    elif   (len(sys.argv) == 2) and (sys.argv[1]=="-a"): 
        print("-----所有參數訓練評估找出最好的參數組合---------")  
        model=evalAllParameter(trainData, validationData,
                        [1, 3, 5, 15, 25], 
                        [10, 50, 100, 200],
                        [0.01, 0.1, 1 ])
    print("==========測試階段===============")
    auc = evaluateModel(model, testData)
    print("使用test Data測試最佳模型,結果 AUC:" + str(auc))
    print("==========預測資料===============")
    PredictData(sc, model)

