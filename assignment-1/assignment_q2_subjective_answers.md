The data is initially split into training (70%) and testing (30%).

The nested 5-fold cross-validation is applied on the training dataset and the average accuracy is reported over validation set and the test dataset. 
The results are as follows:

-------------- Part (a) --------------

Criteria : information_gain
Accuracy:  83.33333333333334
Precision:  0.8571428571428571
Recall:  0.8
Precision:  0.8125
Recall:  0.8666666666666667


-------------- Part (b) --------------

Depth =  1
Average Accuracy for depth  1 Validation Data:  92.85714285714286  | Test Data:  80.66666666666667
===========================================
Depth =  2
Average Accuracy for depth  2 Validation Data:  92.85714285714286  | Test Data:  80.66666666666667
===========================================
Depth =  3
Average Accuracy for depth  3 Validation Data:  92.85714285714286  | Test Data:  83.33333333333334
===========================================
Depth =  4
Average Accuracy for depth  4 Validation Data:  92.85714285714286  | Test Data:  83.33333333333334
===========================================
Depth =  5
Average Accuracy for depth  5 Validation Data:  92.85714285714286  | Test Data:  83.33333333333334
===========================================
Depth =  6
Average Accuracy for depth  6 Validation Data:  92.85714285714286  | Test Data:  83.33333333333334
===========================================
Depth =  7
Average Accuracy for depth  7 Validation Data:  92.85714285714286  | Test Data:  83.33333333333334
===========================================
Depth =  8
Average Accuracy for depth  8 Validation Data:  92.85714285714286  | Test Data:  83.33333333333334
===========================================
Depth =  9
Average Accuracy for depth  9 Validation Data:  92.85714285714286  | Test Data:  83.33333333333334
===========================================

The following dataset can be trained over a depth-1 decision tree itself as no improvement over validation set is seen further.