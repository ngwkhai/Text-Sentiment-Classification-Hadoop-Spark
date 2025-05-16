from pyspark import SparkContext, SparkConf
import pyspark
from pyspark.rdd import RDD
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import time
import re
import sys


class NB:
    def split_csv(self, line):
        columns = line.split(',')
        if len(columns) > 4:
            for i in range(4, len(columns)):
                columns[3] += columns[i]

        return columns

    def clean_text(self, text):
        text = re.sub(
            r"(?i)(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9\-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9\-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
            "", text)
        text = re.sub(r"(#|@|&).*?\w+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^a-zA-Z ]", " ", text)
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text


if __name__ == "__main__":
    NB = NB()
    conf = SparkConf().setAppName("Naive Bayes")
    sc = SparkContext(conf=conf)
    start_time = time.time_ns()
    arg0 = sys.argv[1]

    # Đọc và xử lý dữ liệu
    input = sc.textFile(f"hdfs://mpi6:19000/user/mpi/spark_input_{arg0}/tweets.csv", minPartitions=3) \
        .map(NB.split_csv) \
        .map(lambda column: (
        float(column[1]),  # sentiment
        NB.clean_text(column[3])  # cleaned tweet
    ))

    spark = SparkSession.builder.appName("Naive Bayes").getOrCreate()
    input_dataframe = spark.createDataFrame(input, ["label", "tweet"])

    tokenizer = Tokenizer().setInputCol("tweet").setOutputCol("words")
    words_data = tokenizer.transform(input_dataframe)

    input_hashingTF = HashingTF().setInputCol("words").setOutputCol("rawFeatures")
    input_featurized_data = input_hashingTF.transform(words_data)

    input_idf = IDF().setInputCol("rawFeatures").setOutputCol("features")
    input_idf_model = input_idf.fit(input_featurized_data)

    input_rescaled_data = input_idf_model.transform(input_featurized_data)

    training_data, test_data = input_rescaled_data.randomSplit([0.75, 0.25], seed=1234)

    model = NaiveBayes().fit(training_data)
    predictions = model.transform(test_data)

    end_time = time.time_ns()

    predictions_and_labels = predictions.select("prediction", "label").rdd.map(lambda row: (float(row["prediction"]), float(row["label"])))

    metrics = MulticlassMetrics(predictions_and_labels)

    print(metrics.confusionMatrix)
    print("Accuracy: " + str(metrics.accuracy))
    print("F1 Score: " + str(metrics.weightedFMeasure))
    print("Execution Duration: " + str((end_time - start_time) / 1_000_000_000) + " seconds")

    sc.stop()


