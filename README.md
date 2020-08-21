# Machine-Learning
This is a Repository consisting of many different Machine Learning Models consisting of Linear and Non-Linear Classifiers and trained of datasets to provide someone with knowledge about how to use them .

# Linear Models
### Rosenblatt's Perceptron
It consists of the very initial learning model developed by Rosenblatt's on the principle of gaining knowledge between the difference on the output labels and the target labels but the labels were only considered integers i.e. we used an threshold function even to determine the weights which stopped us from doing any differential on the cost as it would be an non-differential function.
It uses the Iris Flower Dataset to train the model and just makes predictions for a binary classifier using the training data

### Adaline Gradient Descent
This model is an improvement on the previous model as such because it uses an linear function between the weight updates and not an threshold function but uses the threshold function for the output labels and hence we are able to update the weights using gradient descent and hence achieve a better optimum because this makes the cost function differentiable and we could achieve an global optimum using gradient descent.

#### Characteristic about Model
* We have shown the difference that the models have when they are trained with scaled and un-scaled data which shows the importance of feature scaling and that the model also   
  differ in time taken for training as the model with scaled data trains faster in respect with the un-scaled data.
* We have shown the difference in the varied amounts of learning rate which affects model training as the model with large learning rate could overshoot the optimum and hence 
  could diverge while the model which has very small learning rate would take a very long time to reach the minimum. Hence, This shows the importance of having a well defined 
  learning rate which can only be achieved by experimantation.
* Lastly, We showed the basis of Stochastic Gradient Descent also known as Online Learning which shows that when we have huge amount of data we can have faster updates in the 
  weights by using this method but holds a bad characteristic that we might reach the minimum we could oscillate around it due to huge noise in each dataset, hence to decrease 
  this effect we normally use mini batch gradient descent which takes a small batch at a time and runs gradient descent over it which uses the vector implementation hence is 
  faster and also has lesser noise in weight updates also we tend to decrease the learning rate by the increase in the number of iterations.








# References:
## Linear Models:
  ### Rosenblatt's Perceptron:
      https://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf
  ### Adaline Gradient Descent:
      **The paper concerning this concept is currently not available but still wikipedia could give you great information**
      https://en.wikipedia.org/wiki/ADALINE
