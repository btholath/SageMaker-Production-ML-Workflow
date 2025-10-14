"""
The params.py is used by train.py to test different settings for the models. The computer:
Takes the settings from params.py.
Trains models with each combination (like trying alpha=0.1 with fit_intercept=True, then alpha=0.1 with fit_intercept=False).
Saves the results (like how accurate the predictions are) to a tool called MLflow, so you can pick the best model.

Imagine you're in process of buying a house, and you want to make the best choice in buying a house.
You can tweak things like how much price to add, how long to search them, or whether to add or reduce features.
In machine learning, we do something similar with models—we tweak settings to make them predict things better, like guessing house prices.
These settings are called hyperparameters, and the params.py file is like a recipe book that lists different settings to
try for three machine learning models: Ridge, ElasticNet, and XGBoost.

"""

# Hyperparameter grids for model tuning
# Here we have 3 models that predict house prices and each model has its own list of settings to test.
# so computer can try them out and find the best combination.

# Ridge regression parameters
"""
Think of Ridge as a simple way to guess house prices by looking at things like the size of the house or the number of bedrooms. 
It’s a math model that tries to find patterns in the data.
What are these settings?
alpha: 
    This is like a "control knob" that decides how much to simplify the model. 
    A small alpha (like 0.1) lets the model focus on lots of details, while a big alpha (like 10.0) makes it simpler to avoid mistakes. 
    The computer will try 0.1, 1.0, and 10.0 to see which works best.
fit_intercept: 
    This decides if the model should start its predictions from a baseline number (like a starting point on a graph). 
    If True, it uses a baseline; if False, it starts from zero. The computer tries both to find out which is better.
    Example: It’s like deciding how much sugar to add to cookies (a little or a lot) and whether to add a pinch of salt as a starting point.
"""
ridge_param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Regularization strength
    'fit_intercept': [True, False],  # Whether to fit intercept
}




# ElasticNet parameters
"""
ElasticNet is like Ridge, but it’s a bit fancier because it combines two ways of simplifying the model to make better predictions for house prices.

What are these settings?
alpha: 
Same as in Ridge—it controls how much to simplify the model. The computer tries 0.1, 1.0, and 10.0.

l1_ratio: 
    This decides how much to use two different simplification tricks (called L1 and L2). 
    A value of 0.2 means more of the L2 trick (like Ridge), 
    0.8 means more of the L1 trick, and 
    0.5 is a balance. 
    The computer tries all three to find the best mix.
fit_intercept: 
    Same as in Ridge—decides whether to use a baseline number (True) or start from zero (False).
Example: It’s like mixing chocolate and vanilla in different amounts for your cookies and deciding whether to add a pinch of salt.
"""
elasticnet_param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Regularization strength
    'l1_ratio': [0.2, 0.5, 0.8],  # Balance between L1 and L2 regularization
    'fit_intercept': [True, False],  # Whether to fit intercept
}





# XGBoost parameters
"""
What’s XGBoost? XGBoost is like a super-smart robot that builds lots of little decision trees (like flowcharts) to predict house prices.
It’s more complex than Ridge or ElasticNet and often makes better predictions.

What are these settings?
n_estimators: How many decision trees to build. More trees (100, 200, 300) can make better predictions but take longer to train.
learning_rate: How big each step is when the model learns. Smaller steps (0.01) are slower but careful, while bigger steps (0.2) are faster but riskier. The computer tries 0.01, 0.1, and 0.2.
max_depth: How detailed each decision tree can be. A depth of 3 makes simple trees, while 5 makes more complex ones. The computer tries 3, 4, and 5.
min_child_weight: How many data points a tree needs to make a decision. Higher values (1, 2, 3) make the model simpler to avoid mistakes.
subsample: What portion of the data to use for each tree. 0.8 means use 80% of the data, 1.0 means use all of it. The computer tries 0.8, 0.9, and 1.0.
colsample_bytree: What portion of the features (like house size or number of bedrooms) to use for each tree. 0.8 means 80%, 1.0 means all. The computer tries 0.8, 0.9, and 1.0.
gamma: How much improvement a tree needs to make to keep growing. 0 means no minimum, while 0.2 requires more improvement. The computer tries 0, 0.1, and 0.2.
reg_alpha: A trick to simplify the model (L1 regularization). 0 means no simplification, 1.0 means more. The computer tries 0, 0.1, and 1.0.
reg_lambda: Another simplification trick (L2 regularization). 0 means none, 1.0 means more. The computer tries 0, 0.1, and 1.0.



Example: It’s like deciding how many batches of cookies to make, how fast to mix the dough, how many ingredients to use, and how strict to be about the recipe.
"""
xgb_param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
    'max_depth': [3, 4, 5],  # Maximum tree depth
    'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight
    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples for training
    'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features for training
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction for splits
    'reg_alpha': [0, 0.1, 1.0],  # L1 regularization
    'reg_lambda': [0, 0.1, 1.0],  # L2 regularization
}