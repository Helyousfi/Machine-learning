# F1 Score vs ROC AUC vs Accuracy vs PR AUC
## Accuracy :
The accuracy measures how many observations, both positive and negative were correctly classified.
$$ ACC = \frac{TP + TN}{TP + TN + FP + FN} $$
The accuracy shouldn't be used on problems with inbalanced data, because it's easy to get a high accuracy by just classifying all observations as the majority class.  <br>
The accuracy can be used when all the classes are of the same importance to us or when the data is balanced.

## Precision
TP = True positive : the model predicts that the class is positive &  correctly classified <br>
FP = False positive : the model predicts that the class is positive &  miss-classified <br>
TN = True negative : the model predicts that the class is negative & correctly classified <br>
FN = False negative : the model predicts that the class is negative & miss-classified 
$$
precision = \frac{TP}{TP + FP}
$$
Precision refers to the percentage of your results which are relevant.

## Recall
$$
recall = \frac{TP}{TP + FN}
$$
The recall refers to the percentage of total relevant results correctly classified by a model.

## F1 Score
It combines precision and recall in one metric by calculating the harmonic mean between those two. It's a special case of the $F_\beta$ function :
$$
F_\beta = (1+\beta^2)\frac{precision\times recall}{\beta^2 precision + recall}
$$
If we want to care more about precision, we will choose a high beta, otherwise, if we want to care more about recall, we will choose a low beta. <br>
The F1 score could be used in pretty much in every binary classification problem where you care more about the positive class. It is my go-to metric when working on those problems.  

## ROC AUC
AUC means area under the curve so to speak about ROC AUC score we need to define ROC curve first. 
It is a chart that visualizes the tradeoff between true positive rate (TPR) and false positive rate (FPR). Basically, for every threshold, we calculate TPR and FPR and plot it on one chart. <br>
it can be used when we care equally about positive and negative classes.

