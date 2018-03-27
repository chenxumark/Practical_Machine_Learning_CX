Predicting Exercise Quality With Machine Learning
-------------------------------------------------

Date: March 27, 2018<br/> E-mail: <chenxumark@gmail.com><br/> Author: Chen Xu<br/>

### Executive Summary

People regularly quantify how much of a particular activity they do, but they rarely quantify how well they do it. Therefore, the goal of this report is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict if an exercise is correctly performed or not.

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

This project will utilize machine learning techniques to predict which class of exercise fits best the given test dataset, composed by 20 observations.

### Exploratory Analysis

The data of exploratory analysis can be downloaded online and the process can be done as below.

``` r
train_data_url  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"; 
test_data_url   <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

train_data_file <- "training_data.csv"
test_data_file  <- "testing_data.csv"

if(!file.exists(train_data_file)) 
        download.file(train_data_url, destfile = train_data_file)
if(!file.exists(test_data_file))
        download.file(test_data_url,  destfile = test_data_file)

# Weight Lifting Exercise dataset
train_data<- read.csv(train_data_file)
test_data<- read.csv(test_data_file)

d <- dim(train_data)
```

The raw training data contains 19622 observations of 160 variables.

### Data Cleaning

In this step, the dataset will be cleaned and get rid of observations with missing values as well as some meaningless variables.

1.  clean the Near Zero Variance Variables.

``` r
NZV <- nearZeroVar(train_data, saveMetrics = TRUE)
train_data<- train_data[, !NZV$nzv]
test_data<- test_data[, !NZV$nzv]
```

1.  remove some columns of the dataset that do not contribute much to the accelerometer measurements.

``` r
regex <- grepl("^X|timestamp|user_name", names(train_data))
training <- train_data[, !regex]
testing <- test_data[, !regex]
```

3.Removing columns that contain NA's

``` r
cond <- (colSums(is.na(training)) == 0)
training <- training[, cond]
testing <- testing[, cond]
d1<-dim(training)
d2<-dim(testing)
```

After cleaning process, we are left with 19622 observations of 54 variables to build our model.

4.Correlation Matrix of Columns in the Training Data set.

``` r
corrplot(cor(training[, -length(names(training))]), method = "color", tl.cex = 0.5, tl.col="gray")
```

<img src="Project_files/figure-markdown_github/pre.process.corr-1.png" style="display:block; margin: auto" style="display: block; margin: auto;" />

### Model Training

Dataset can be split into `train_data` & `test_data` and then proceed with the exploratory analysis.

``` r
set.seed(1234) # To be reproducible

intrain<- createDataPartition(training$classe, p=0.75, list=FALSE)
training <- training[intrain,]
validation  <- training[-intrain,]
```

Now that we have the most significant variables selected, we will setup the parallel processing libraries using the method described by Greski. This will allow faster processing times on multi-core CPUs.

``` r
cluster <- makeCluster(detectCores()) # I'm running in a VM so I'm using all cores
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)
```

Next, three different models will be fit to check which one is more accurate for this dataset: random forest (rf), boosting with trees (gbm) and linear discriminant analysis (lda).

``` r
model1 <- train(classe ~ ., method = "rf",  data = training, trControl = fitControl, verbose = FALSE)
model2 <- train(classe ~ ., method = "gbm", data = training, trControl = fitControl, verbose = FALSE)
model3 <- train(classe ~ ., method = "lda", data = training, trControl = fitControl, verbose = FALSE)

stopCluster(cluster) # close parallel cluster
```

The code below will run the predictions for the `validation` dataset using the models built above and plot the confusion matrices and accuracies for each model.

``` r
p1 <- predict(model1, validation)
p2 <- predict(model2, validation) 
p3 <- predict(model3, validation)

c1 <- confusionMatrix(p1, validation$classe)
c2 <- confusionMatrix(p2, validation$classe)
c3 <- confusionMatrix(p3, validation$classe)


c1 #random forest (rf) 
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1046    0    0    0    0
    ##          B    0  711    0    0    0
    ##          C    0    0  642    0    0
    ##          D    0    0    0  615    0
    ##          E    0    0    0    0  693
    ## 
    ## Overall Statistics
    ##                                     
    ##                Accuracy : 1         
    ##                  95% CI : (0.999, 1)
    ##     No Information Rate : 0.2822    
    ##     P-Value [Acc > NIR] : < 2.2e-16 
    ##                                     
    ##                   Kappa : 1         
    ##  Mcnemar's Test P-Value : NA        
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2822   0.1918   0.1732   0.1659   0.1869
    ## Detection Rate         0.2822   0.1918   0.1732   0.1659   0.1869
    ## Detection Prevalence   0.2822   0.1918   0.1732   0.1659   0.1869
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

``` r
c2 #boosting with trees (gbm) 
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1044    1    0    0    1
    ##          B    2  706    3    1    2
    ##          C    0    4  637    6    1
    ##          D    0    0    2  608    4
    ##          E    0    0    0    0  685
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9927          
    ##                  95% CI : (0.9894, 0.9952)
    ##     No Information Rate : 0.2822          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9908          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9981   0.9930   0.9922   0.9886   0.9885
    ## Specificity            0.9992   0.9973   0.9964   0.9981   1.0000
    ## Pos Pred Value         0.9981   0.9888   0.9830   0.9902   1.0000
    ## Neg Pred Value         0.9992   0.9983   0.9984   0.9977   0.9974
    ## Prevalence             0.2822   0.1918   0.1732   0.1659   0.1869
    ## Detection Rate         0.2816   0.1905   0.1718   0.1640   0.1848
    ## Detection Prevalence   0.2822   0.1926   0.1748   0.1656   0.1848
    ## Balanced Accuracy      0.9987   0.9951   0.9943   0.9933   0.9942

``` r
c3 #linear discriminant analysis (lda)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   A   B   C   D   E
    ##          A 866  92  56  42  15
    ##          B  31 478  62  23 106
    ##          C  67  81 441  73  65
    ##          D  80  31  67 458  61
    ##          E   2  29  16  19 446
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7254          
    ##                  95% CI : (0.7107, 0.7397)
    ##     No Information Rate : 0.2822          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6529          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8279   0.6723   0.6869   0.7447   0.6436
    ## Specificity            0.9230   0.9259   0.9067   0.9227   0.9781
    ## Pos Pred Value         0.8086   0.6829   0.6066   0.6571   0.8711
    ## Neg Pred Value         0.9317   0.9225   0.9326   0.9478   0.9227
    ## Prevalence             0.2822   0.1918   0.1732   0.1659   0.1869
    ## Detection Rate         0.2336   0.1289   0.1190   0.1236   0.1203
    ## Detection Prevalence   0.2889   0.1888   0.1961   0.1880   0.1381
    ## Balanced Accuracy      0.8754   0.7991   0.7968   0.8337   0.8108

The random forest model outperform both the gbm and lda models, the lda being by far the one with the worst performance.

### Predicting the test Set

The final step is to predict the testing data set composed by 20 observations, as proposed by the project's especification. We will use the random forest model above, since it gave the best predictions.

``` r
pred <- predict(model1, testing)
pred
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

### Conclusion

For this Report, the random forest model (`rf`) outperformed both the boosting with trees (`gbm`) and linear discriminant analysis (`lda`) models. Nevertheless, the `gbm` model performed close to the random forest, suggesting that it could be applicable to a real world scenario.
