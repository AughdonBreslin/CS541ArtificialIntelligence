### Aughdon Breslin

### CS 541 Artificial Intelligence - Jie Shen

# Final Cheat Sheet

#### 1. Knowledge on Recommendation Systems

**How is machine learning used in Recommendation Systems?**

Recommendation systems are a type of machine learning algorithm that are used to make personalized suggestions to users based on their interests or preferences. These systems can be used in a variety of applications, such as recommending products to customers in an online store, suggesting content to users on a social media platform, or providing personalized learning experiences for students. To use recommendation systems in machine learning, one common approach is to train a model on a large dataset of user preferences or interactions, such as the items that users have purchased or the content that they have interacted with in the past. The model can then use this information to make predictions about what a given user is likely to be interested in, and provide personalized recommendations accordingly. These predictions can be refined and improved over time as the model receives more data and feedback from users.

**What is the tradeoff between binary and continuous Recommendation Systems?**

The tradeoffs between binary and continuous recommendation systems refer to the differences in the way that these systems make recommendations to users. Binary recommendation systems, as the name suggests, make recommendations by classifying items into two categories: "like" or "dislike." This can be useful in some applications, such as providing a simple yes/no recommendation to a user. However, binary recommendation systems do not provide as much flexibility or granularity as continuous recommendation systems, which can make more nuanced recommendations by assigning a numerical score or rating to each item. Continuous recommendation systems can provide more personalized and detailed recommendations, but may require more complex models and larger amounts of training data. Ultimately, the choice between binary and continuous recommendation systems will depend on the specific requirements and goals of the application.

#### 2. Knowledge on Noise/Robustness

##### What is convex programming?

Convex programming is a mathematical optimization technique used to solve optimization problems that have a convex objective function and convex constraints. In other words, it is a way of finding the optimal values for a set of variables that minimize or maximize a specific objective function subject to certain constraints.

Convex programming is based on the principles of convex analysis, which is a branch of mathematics that studies the properties of convex sets and convex functions. A convex function is a function that has the property of being "well-behaved" in the sense that it is always above or on its own tangent lines. This property allows for the use of certain mathematical techniques to solve optimization problems involving convex functions more efficiently than with other types of optimization techniques.

Some common applications of convex programming include portfolio optimization in finance, training of machine learning models, and power system optimization in electrical engineering. Convex programming has the advantage of providing global optimal solutions to optimization problems, rather than just locally optimal solutions. It is also relatively fast and efficient compared to other optimization techniques.

**How can we reduce noise in a dataset?**

There are several ways to reduce noise in a dataset. One approach is to use **data cleaning and preprocessing** techniques to remove or correct inaccuracies or inconsistencies in the data. This can involve identifying and removing duplicate or out-of-range data points, imputing missing values, or applying transformations to the data to make it more consistent.

Another approach to reducing noise in a dataset is to use **dimensionality reduction** techniques, such as principal component analysis or singular value decomposition. These techniques can help to identify and remove redundant or irrelevant features in the data, which can reduce the amount of noise and improve the performance of machine learning algorithms.

Additionally, using **regularization** techniques during model training can help to reduce overfitting and prevent the model from learning noise in the data. This can involve adding constraints or penalties to the model's objective function, which can prevent the model from fitting to random variations in the data and improve its generalization performance.

Overall, the best approach to reducing noise in a dataset will depend on the specific characteristics of the data and the goals of the analysis. A combination of data preprocessing, dimensionality reduction, and regularization techniques can often provide good results in reducing noise and improving the quality of the data.

**How can we improve robustness in a machine learning model?**

There are several ways to improve the robustness of a machine learning model. One approach is to use **regularization** techniques during training, which can prevent the model from overfitting to the training data and improve its ability to generalize to new, unseen data. This can involve adding constraints or penalties to the model's objective function, which can help to reduce the model's sensitivity to random variations in the data and improve its performance on out-of-sample data.

Another way to improve the robustness of a machine learning model is to use **data augmentation** techniques, which can expand the size and diversity of the training dataset. This can involve applying transformations to the existing data, such as rotating, scaling, or cropping images, or adding random noise to the data, in order to create additional, synthetic training examples. Using a **larger and more diverse training dataset** can help to improve the model's robustness by providing it with more information about the underlying patterns and structures in the data.

Additionally, using **ensembling methods, which combine the predictions of multiple models**, can also improve the robustness of a machine learning system. Ensembling can help to reduce the overall variance of the predictions, and can provide more stable and reliable results.

Overall, there are many different approaches that can be used to improve the robustness of a machine learning model, and the best approach will depend on the specific characteristics of the data and the goals of the analysis.

#### 3. General Question - Possibly Mean Estimation Against an Adversary

**What is Maximum Likelihood Estimation?**

Maximum likelihood estimation (MLE) is a method for estimating the parameters of a statistical model. Given a set of observations and a statistical model, MLE finds the set of parameters that maximizes the likelihood function, which is a function that describes the probability of the observed data given the model and its parameters. In other words, MLE determines the values of the parameters that make the observed data most probable, assuming that the model is correct. This approach can be used to estimate the parameters of a wide variety of statistical models, including linear regression, logistic regression, and many others.

**How can MLE be used to estimate the mean even if the dataset contains "dirty" data?**

If you have a dataset that contains some "dirty" or missing data, you can use MLE to estimate the mean of the dataset. The idea is to use the observed data to construct a likelihood function, which describes the probability of the observed data given the model and its parameters. In this case, the model would be a normal distribution with unknown mean and variance, and the parameters would be the mean and variance of the distribution.

To use MLE to estimate the mean, you would first need to specify a model for the data. For example, you might assume that the data comes from a normal distribution with unknown mean and variance. Then, you would use the observed data to calculate the likelihood function, which is a function that describes the probability of the observed data given the model and its parameters.

Next, you would need to maximize the likelihood function with respect to the mean. This would give you the maximum likelihood estimate of the mean. This estimate would be the value of the mean that makes the observed data most probable, assuming that the model is correct.

Once you have estimated the mean using MLE, you can use it to impute missing values in the dataset. For example, if some of the data values are missing, you can use the MLE estimate of the mean to fill in the missing values, assuming that the data follows the normal distribution with the estimated mean and variance. This can help you to clean up the dataset and make it more useful for further analysis.

#### 4. Knowledge on Dimensional Reduction

##### SVD in PCA

In principal component analysis (PCA), singular value decomposition (SVD) is a method used to decompose a matrix into its constituent parts. SVD is used in PCA to decompose the covariance matrix of the data into its eigenvectors (principal components) and eigenvalues.

SVD is a powerful mathematical tool that allows a matrix to be decomposed into the product of three simpler matrices. In the context of PCA, the original matrix is the covariance matrix of the data, which contains information about the correlations between the different variables in the dataset. SVD decomposes this matrix into the product of two matrices and their transpose.

The first matrix, called the left singular matrix, contains the eigenvectors of the covariance matrix, which are also known as the principal components. These eigenvectors capture the underlying structure of the data and are used to project the data onto a lower-dimensional space.

The second matrix, called the diagonal matrix, contains the eigenvalues of the covariance matrix, which indicate the amount of variation in the data explained by each principal component.

The third matrix, called the right singular matrix, is the transpose of the left singular matrix and contains the same information as the left singular matrix.

In summary, SVD is a mathematical method used in PCA to decompose the covariance matrix of the data into its constituent parts. The resulting eigenvectors and eigenvalues are used to identify the principal components of the data and to project the data onto a lower-dimensional space.

##### Tradeoff Between RP and PCA

The tradeoff between random projection and principal component analysis (PCA) lies in the amount of computational time and memory required to perform the techniques, as well as the quality of the results.

Random projection is a method for dimensionality reduction that involves projecting the data onto a lower-dimensional space using a randomly generated matrix. This method is fast and efficient, and it can be used to reduce the dimensionality of very large datasets. However, the results of random projection may not be as good as those of PCA, as the random matrix used for projection may not capture the underlying structure of the data as well as the principal components generated by PCA.

PCA, on the other hand, is a more computationally intensive method for dimensionality reduction that involves computing the eigenvectors of the covariance matrix of the data. These eigenvectors, known as principal components, capture the underlying structure of the data and can be used to reduce the dimensionality of the dataset while preserving as much of the variation in the data as possible. The results of PCA are generally of higher quality than those of random projection, but the computational time and memory required to perform PCA can be prohibitive for very large datasets.

In summary, the tradeoff between random projection and PCA is the balance between computation time and memory requirements, and the quality of the results. Random projection is faster and more efficient, but the results may not be as good as those of PCA. PCA, on the other hand, produces higher-quality results but is more computationally intensive.

#### 5. Knowledge on Active Learning

##### How does active learning work?

Active learning is a type of machine learning that involves training a model on a small initial dataset, and then iteratively selecting the most informative data points to be labeled and added to the training set. The goal of active learning is to improve the performance of the model by selecting the most valuable data for training, rather than using a large, randomly selected dataset.

Active learning typically involves the following steps:

1. **Initialize the model with a small, randomly selected dataset.**
2. **Use the model to make predictions on a larger, unlabeled dataset.**
3. **Select the data points that are most informative or uncertain according to the model's predictions, and label them.**
4. Add the newly labeled data points to the training set and retrain the model.
5. Repeat steps 2-4 until the model reaches a satisfactory level of performance.

One advantage of active learning is that it can reduce the amount of labeled data required to train a high-performing model. This is especially useful in cases where labeling data is expensive or time-consuming, as it allows the model to be trained on a smaller, more focused dataset. Another advantage is that active learning can improve the performance of the model by selecting the most valuable data for training. By focusing on the data points that are most informative or uncertain, active learning can help the model to learn more effectively and make better predictions.

##### What is an example for which active learning would not work?

One example where active learning may not work is in cases where the data is not representative of the underlying population. For example, suppose we are using active learning to train a machine learning model to classify images of animals. If the initial training dataset only contains images of cats and dogs, the model may not be able to accurately classify other types of animals, such as birds or horses. In this case, even if the active learning algorithm selects the most uncertain or informative images for labeling, the model's performance may not improve because **the data does not represent the full range of classes in the underlying population.**

In general, active learning is most effective when the initial training dataset is representative of the underlying population and when the data is sufficiently diverse and informative. If the initial training dataset is not representative or if the data is not sufficiently diverse, active learning may not be able to improve the performance of the model.