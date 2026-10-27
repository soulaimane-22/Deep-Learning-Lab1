# Personal Synthesis

In this lab, I learned how to build and train MLP models with PyTorch for regression and multiclass classification. I practiced exploring data, scaling numerical features, encoding categories, and choosing the correct output layer and loss function for each task.

I learned that a more complex model is not always better. For AAPL, simply predicting that the next closing price would equal the current price performed better than all the tested neural networks. This showed me why a simple baseline is important.

For classification, I saw why accuracy alone can be misleading. Most records were No Failure, so a model could have high accuracy while missing rare failures. Macro-F1, recall, and confusion matrices helped me see these weaknesses. Oversampling detected more failures, but also produced more false alarms.

GridSearchCV helped me compare learning rates, optimizers, epochs, and architectures. Regularization did not improve every result: L2 reduced validation loss, while the Reference model had the best validation macro-F1. The learning curves also helped me recognize overfitting when training loss decreased but validation loss increased.

Finally, I learned to fit preprocessing only on training data and use validation results to select models before evaluating the test set. For stock prices, chronological splits keep future observations out of training. Early stopping scored better on the classification test set, but I kept the model selected using validation. These results come from one seed and split, and the rare classes contain few examples, so the conclusions remain limited. Future work could test several seeds and use more rare-failure examples to check whether the findings hold.
