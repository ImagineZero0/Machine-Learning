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

# Scikit-Learn Models
It consists of many different models which are implemented by using the Scikit Learn Library. This folder also reflects the need for having many different models to train or use for the same problem only because their cannot be any model which specializes in one particular problem and then also in performs very well in any other problem as it is stated by the "No Free Lunch Theorem". If you want to read more about it, you can refer to the links below.
## Parametric and Non-Parametric Models:
Here we also have trained different models on the basis of there having any particular trained parameters or weights and not having them and hence they can be termed as parametric and non-parametric models.
### Perceptron Model:
This is the same model that we had trained before by implementing explicitly but here we have used the scikit learn library and hence trained the model on the provided dataset and also tuned the parameters to have better training accuracy. We have also plotted the decision boundary and hence verified the model's accuracy. The advantage of using the perceptron model from the scikit learn library is that it also implement's the OvR technique i.e. One-vs-All technique of classfication and hence we can use this model for classification.

### Logistic Regression:
This is a better model wrt the previous model as this model implements the weight updates by also using the real penalties and not only integer penalties and hence we could have better weight updates, We achieve this penalty update by using the sigmoid function and hence using it in the cost function which makes it a differentiable and convex function and hence we are able to update weights more efficiently and achieve a better optimum value, and by using this penalty measure we could update the weights in a better way because of this we can alter the weights in a amount proportional to the penalties and hence update them accordingly as opposed to how we used to do it previously.
We have also both explicitly implemented an Logistic regression class and also using the scikit learn package and showed the different parameter tuning and also how was this model different wrt the Adaline model we implemented.
In this model we typically access the model's probability and measaure the log of it to access the data's probabilties of being in a particular class.

### Support Vector Machines:
This is a comparitively more better modeled in some cases wrt the logistic regression model as it captures the models classification in the best way possible by taking the data points as support vector for making a decision boundary between different classifiers and hence tries to identify a decision boundary between the support vector's for best classification and we can try to have different model boundaries depending upon the model parameters we provide it.We try to achieve a maximum margin for better model classification.The basic terminlogy that we use for this case would be soft margin classificatin by which we change a parameter and hence obtain different decision boundaries as per our choices.
We can also use Support Vector Machines for better generalization on Non-Linear Models which have decision boundaries not linear and cannot be classified on that basis and hence we use the method of kernels on Support Vector Machines to achieve this non linearity model.The basic idea is that we transform out small dimensional dataset into high dimensoins and so that we could get an hyperplane to classify the dataset and then project this hyperplane onto the smaller dimension and hence we achieve a non linear decision boundary for the dataset.Most readily used kernel is the radial basis function also known as the gaussian kernel and it measures the similarity between different data points. Therefore, using this approach we are able to calssify the data into for non linear boundaries.

### Decision Tree:
These are one of the most widely used models for ensemble learning which are used both in academia and industries and the basic idea behind them is to quantify the data between different nodes and achieve a better interpretability of the dataset and we do this by maximizing the information gain at each node by taking different ways of measuring the impurities like the Gini Impurity, entropy and Classification Error.Their usage is enhanced when we combine multiple decision trees to build random forests and hence achieve better results and generalize better.

### K - Nearest Neighbors:
This is a type of non parametric model and commonly known as "lazy learner",because of its nature not to capture the data's pattern but to learn the data's datapoints and hence predict the new datapoints according to that.They work on the basis that we choose a particular amount of neighbors around a datapoint and hence choose the type of datapoint nature from the datapoint in it's neighbourhood. The basis through which we choose the datapoint is based upon the distance metric which can be from Euclidean Distance, Machattant Distance and many other generalizations of them.

### Characteristics:
* From this module we have also started to preprocess the data by using the scikit learn varied modules to have the dataset as required for training the required models.
* We learnt about the different models that can be implemented using the scikit learn library and how well we can tune the parameters.
* We also tackled the problem of overfitting and how to control it using regularrization, We mostly tackle overfitting when we are using a very complex model for training ono a 
  dataset to capture the datasets patterns but in turn end up capturing also the dataset's noise, hence we use regularization for better generalizaition of model.
* We have also defined the different types of parametric and non parmetric models like Logistic Regression and Support Vector Machines and also K - Nearest Neighors and Ensemble 
  Models like Decsion Trees and Random Forests.
* We have also implemented the different types characteristics for each model like the regularization parameter for logistic regression and sigmoid function and also the measure 
  of Information Gain.Also we have used different libraries to visalize the decision tree models for training.
* The last thing that we get to address from this is the Curse of Dimensionality that mostly affects all type of models because of its nature to make the dataset sparse even 
  when we have a huge dataset because of the increased distance in them. 

# References:
  ### Rosenblatt's Perceptron:
      https://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf
  ### Adaline Gradient Descent:
      **The paper concerning this concept is currently not available but still wikipedia could give you great information**
      https://en.wikipedia.org/wiki/ADALINE
  ### No-Freer Lunch Theorem
      https://www.researchgate.net/publication/2755783_The_Lack_of_A_Priori_Distinctions_Between_Learning_Algorithms
  ### Logistic Regression:
      https://methods.sagepub.com/book/logistic-regression-from-introductory-to-advanced-concepts-and-applications
  ### Support Vector Machines:
      https://b-ok.asia/book/2481060/114922?regionChanged=&redirect=7692167
      https://www.di.ens.fr/~mallat/papiers/svmtutorial.pdf
  ### Decision Trees:
      https://en.wikipedia.org/wiki/Decision_tree
  ### K-Nearest Neighbors:
      https://www.researchgate.net/publication/220493118_An_Algorithm_for_Finding_Best_Matches_in_Logarithmic_Expected_Time
