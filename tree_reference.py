## Bagging
One of the problems of decision trees is the high variance. Bootstrap offers one solution to reduce variance.

1. Take B many bootstrap samples
2. One tree for each B sample
3. The final prediction = Average of all B  trees
4. It is a special case for Random Forest

**Pros**: Reduce the var while maintain similar bias.

**Cons**: The trees are correlated (on higher level) because the decision trees are prone to first split the strong predictors.

## Random forest
On one hand we want to reduce the variance by bagging; on the other hand, bagging produces highly correlated trees which turns out will not help reduce variance substantially. Random forest tackles this problem by forcing to split only a subset of predictors.

1) Take B many bootstrap samples
2) Build a deep random tree for each bootstrap sample by splitting only m (`mtry`) randomly chosen predictors at each split
3) Bag all the random trees by taking average => prediction of y given $x_1,...,x_p$
  4) Use Out of Bag testing errors to tune `mtry`


**Pros**: Decorrelate the trees - reduce var more

**Cons**: Tuning parameter `m` is introduced

- m too small: miss important var's
- m too large: more cor between trees


Remark:

1) Nodesize: 5 for regression trees and 1 for classification trees

2) When mtry = p (= 19), randomForest gives us bagging estimates.



Default settings:

- Mtry = p/3, ($\sqrt{p}$ in classification tree)
- Bootstrap size B = ntree = 500

And as noted above:

- when mtry = p = 19, randomForest gives us bagging estimates.
- nodesize:5 for reg's and 1 for classifications.





#### b) OOB MSE
After we get the OOB predicted value for each observation, then we can calculate the OOB MSE. 

```{r}
plot(fit.rf$mse, xlab="number of trees", col="blue",
     ylab="ave mse up to i many trees using OOB predicted",
     pch=16) # We only need about 100 trees for this
```

We get the above plot also by
```{r}
plot(fit.rf, type="p", pch=16,col="blue" )
```

Compare the MSE for training data and OOB
```{r}
mse.train <- mean((data1$LogSalary-yhat)^2)
mse.oob <- mean((data1$LogSalary-fit.rf$predicted)^2)  #OOB testing error
```

The training MSE is `r mse.train` and the OOB MSE is `r mse.oob`. 


### iii) The effect of changing `ntree` and `mtry`

Ready to tune mtry and B=number of the trees in the bag

#### a) ntree: given mtry, we see the effect of ntree first
```{r}
fit.rf <- randomForest(LogSalary~., data1, mtry=10, ntree=500)
plot(fit.rf, col="red", pch=16, type="p", main="default plot")
```
We may need 250 trees to settle the OOB testing errors

#### b) mtry: the number of random split at each leaf

Now we fix `ntree=250`, We only want to compare the OOB mse[250] to see the mtry effects.
Here we loop mtry from 1 to 19 and return the testing OOB errors

```{r}
rf.error.p <- 1:19  # set up a vector of length 19
for (p in 1:19)  # repeat the following code inside { } 19 times
{
  fit.rf <- randomForest(LogSalary~., data1, mtry=p, ntree=250)
  #plot(fit.rf, col= p, lwd = 3)
  rf.error.p[p] <- fit.rf$mse[250]  # collecting oob mse based on 250 trees
}
rf.error.p   # oob mse returned: should be a vector of 19

plot(1:19, rf.error.p, pch=16,
     xlab="mtry",
     ylab="OOB mse of mtry")
lines(1:19, rf.error.p)
```

Run above loop a few time, it is not very unstable.
Notice

1. mtry = 1 is clearly not a good choice.
2. The recommended mtry for reg trees are mtry=p/3=19/3 about 6 or 7. Seems to agree with this example.  Are you convinced with p/3?
  
  We should treat mtry to be a tuning parameter!
  
  The final fit: we take mtry = 6
```{r}
fit.rf.final <- randomForest(LogSalary~., data1, mtry=6, ntree=250)
plot(fit.rf.final)
```
