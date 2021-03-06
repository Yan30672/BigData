# -*- coding: UTF-8 -*-
import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import  MatrixFactorizationModel
  
def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("Recommend")                         \
                         .set("spark.ui.showConsoleProgress", "false") \
               
    sc = SparkContext(conf = sparkConf)
    print("master="+sc.master)
    SetLogger(sc)
    SetPath(sc)
    return (sc)

def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="file:/home/hduser/pythonsparkexample/PythonProject/"
    else:   
        Path="hdfs://master:9000/user/hduser/"

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    
def PrepareData(sc): 
    print("開始讀取電影ID與名稱對照表...")
    itemRDD = sc.textFile(Path+"data/u.item") 
    movieTitle= itemRDD.map( lambda line : line.split("|"))     \
                                   .map(lambda a: (float(a[0]),a[1]))       \
                                   .collectAsMap()                          
    return(movieTitle)

def RecommendMovies(model, movieTitle, inputUserID): 
    RecommendMovie = model.recommendProducts(inputUserID, 10) 
    print("針對使用者id" + str(inputUserID) + "推薦下列電影:")
    for rmd in RecommendMovie:
        print  "針對使用者id {0} 推薦電影{1} 推薦評分 {2}". \
            format( rmd[0],movieTitle[rmd[1]],rmd[2])

def RecommendUsers(model, movieTitle, inputMovieID) :
    RecommendUser = model.recommendUsers(inputMovieID, 10) 
    print "針對電影 id {0} 電影名:{1}推薦下列使用者id:". \
           format( inputMovieID,movieTitle[inputMovieID])
    for rmd in RecommendUser:
        print  "針對使用者id {0}  推薦評分 {1}".format( rmd[0],rmd[2])


def loadModel(sc):
    try:        
        model = MatrixFactorizationModel.load(sc, Path+"ALSmodel")
        print "載入ALSModel模型"
    except Exception:
        print "找不到ALSModel模型,請先訓練"
    return model 



def Recommend(model):
    if sys.argv[1]=="--U":
        RecommendMovies(model, movieTitle,int(sys.argv[2]))
    if sys.argv[1]=="--M": 
        RecommendUsers(model, movieTitle,int(sys.argv[2]))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("請輸入2個參數")
        exit(-1)
    sc=CreateSparkContext()
    print("==========資料準備===============")
    (movieTitle) = PrepareData(sc)
    print("==========載入模型===============")
    model=loadModel(sc)
    print("==========進行推薦===============")
    Recommend(model)

    

