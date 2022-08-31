from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
import os
from dotenv import load_dotenv

load_dotenv()

spark_master = os.getenv("master", "local[*]")
spark_name = os.getenv("spark_name", "no_env_file")

conf = SparkConf()
conf.setMaster(spark_master).setAppName(spark_name)

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
