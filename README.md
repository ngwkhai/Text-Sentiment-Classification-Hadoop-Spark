# Ứng dụng Naive Bayes và SVM trong phân tích cảm xúc văn bản lớn trên nền tảng Hadoop và Spark

Mã nguồn được phát triển phục vụ cho dự án bài tập lớn có cùng tên, dưới sự hướng dẫn của giảng viên hướng dẫn tại viện Trí tuệ nhân tạo của trường Đại học Công nghệ.


## Mục tiêu chính

* Nghiên cứu các kiến thức nền tảng về khai phá văn bản, phân tích cảm xúc, MapReduce và tính toán song song/phân tán.
* Xây dựng các ứng dụng phân tích cảm xúc sử dụng hai framework: **Apache Hadoop** và **Apache Spark**.
* Đánh giá hiệu năng mô hình qua các chỉ số như **độ chính xác, F1-score**, và khả năng song song hóa qua **thời gian thực thi, tốc độ xử lý và khả năng mở rộng**.
* Đưa ra kết luận về hiệu suất của từng mô hình và đề xuất các hướng mở rộng trong tương lai.

---

## Các ứng dụng đã phát triển

Tất cả các mô hình phân loại đều sử dụng **75% dữ liệu để huấn luyện** và **25% để kiểm tra**.

### Trên nền tảng Hadoop

* [Phiên bản của Naive Bayes](Hadoop/NB.java)
* [Phiên bản của SVM](Hadoop/SVM.java)


### Trên nền tảng Spark

* [Phiên bản của Naive Bayes](Spark/NaiveBayes/python/NaiveBayes.py)
* [Phiên bản của SVM](Spark/SVM/python/SVM.py)

---

## Dữ liệu đầu vào

Tập dữ liệu gồm **1.6 triệu tweet**, ở định dạng `.csv`, mỗi dòng gồm ID, nhãn cảm xúc, và nội dung tweet, ví dụ:

```
0,Sentiment140,I must think about positive..
```

Dữ liệu đầu vào được chia thành 10 tập con có kích thước từ **100.000 đến 1.000.000 tweet**, được lưu tại thư mục [`/input`](input/).

* Đối với Hadoop: các file có tên `train#` và `test#` (`#` từ 1 đến 10).
* Đối với Spark: các file có tên `spark_input_#` (`#` từ 1 đến 10).

---

## Hướng dẫn thực thi

###  Môi trường phát triển

* **Apache Hadoop**
* **Apache Spark**
* **Java**
* **Python**


### Clone dự án

```bash
git clone https://github.com/ngwkhai/Text-Sentiment-Classification-Hadoop-Spark.git
cd Text-Sentiment-Classification-Hadoop-Spark
```



### Khởi động Hadoop

```bash
start-dfs.sh
start-yarn.sh
```

### Upload dữ liệu vào HDFS

```bash
hdfs dfs -mkdir -p /user/username/input_text_sentiment
hdfs dfs -put /input /user/username/input_text_sentiment/
```
### Thực thi trên Hadoop

#### Với Naive Bayes

```bash
cd Hadoop/NB
mkdir NB_classes
```
```bash
javac -classpath "$(yarn classpath)" -d NB_classes NB.java
jar -cvf NB.jar -C NB_classes/ .
hadoop jar NB.jar NB /input_text_sentiment/train# /input_text_sentiment/test# training_split testing_split
```


#### Với SVM

```bash
cd Hadoop/SVM
mkdir SVM_classes
```

```bash
javac -classpath "$(yarn classpath)" -d SVM_classes SVM.java
jar -cvf SVM.jar -C SVM_classes/ .
hadoop jar SVM.jar SVM /input_text_sentiment/train# /input_text_sentiment/test# training_split testing_split
```


**Chú thích**:

* `train#`, `test#`: tên các file dữ liệu trong thư mục `/input`.
* `training_split`, `testing_split`: số byte để chia nhỏ dữ liệu cho các mapper.

---

### Thực thi trên Spark

#### Với Naive Bayes
```bash
spark-submit --master yarn /path/to/NaiveBayes.py {arg0}
```

#### Với SVM
```bash
spark-submit --master yarn /path/to/SVM.py {arg0}
```

**Chú thích**:

* `/path/to/NaiveBayes.py`, `/path/to/SVM.py`: Đường dẫn tuyệt đối tới file python của thuật toán.
* `arg0`: Tham số đầu tiên trong mảng tham số, biểu thị số thứ tự của file input cần chạy (`spark_input_{arg0}`).

##  Kết quả và đánh giá

Kết quả phân loại được phân tích theo:

* **Ma trận nhầm lẫn (confusion matrix)**
* **Chỉ số F1-score, độ chính xác (accuracy)**
* **Tốc độ thực thi, khả năng mở rộng (scalability) và speedup**

---

