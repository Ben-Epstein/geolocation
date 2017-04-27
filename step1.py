pyspark
sc
mydata = sc.textFile("file:/home/training/training_materials/data/purplecow.txt")
mydata2 = sc.textFile("frostroad.txt")
mydata.count()
mydata2.count()
mydata.collect()
mydata2.collect()


#Jobs are run typically locally on a single core
#Adding tags like --master yarn-client or --master yarn-cluster will change where it is run


spark-submit wordcount.py frostroad.txt
spark-submit wordcoiunt.py /home/training/training_materials/data/purplecow.txt

spark-submit --master yarn-cluster wordcount.py frostroad.txt
