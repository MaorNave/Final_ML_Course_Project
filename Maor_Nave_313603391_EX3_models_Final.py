
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor






models_ex1 =  {
    'Logistic_Regression': (LogisticRegression(random_state=42), {
        'C': [ 6],  # Regularization parameter (inverse of strength)
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],  # Optimization algorithm
        'max_iter': [10]
    }),
    'Random_Forest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [ 7000, 8000, 9000, 10000, 15000, 20000],  # Number of trees in the forest
        'max_depth': [16, 17, 18, 19, 20, 21],  # Maximum depth of individual trees
        'min_samples_split': [2],  # Minimum samples required to split a node
        'min_samples_leaf': [1]  # Minimum samples required at each leaf node
    }),
    'SVM': (SVC(random_state=42), {
        'C': [ 8 ,9],  # Regularization parameter (penalty for model complexity)
        'kernel': ['linear', 'rbf'],  # Kernel function (linear or radial basis function)
        'gamma': [0.1],  # Kernel coefficient for RBF kernel
    }),
    'XGBoost': (xgb.XGBClassifier(random_state=42), {
        'n_estimators': [400],  # Number of boosting stages
        'max_depth': [8],  # Maximum depth of individual trees
        'learning_rate': [ 0.12],  # Step size in each boosting step
        'gamma': [0, 0.001, 0.003],  # Minimum loss reduction required for a split
    }),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {
        'n_estimators': [1500],  # Number of boosting stages
        'learning_rate': [0.3],  # Step size in each boosting step
        'algorithm': ['SAMME.R', 'SAMME']  # Base learner algorithm
    }),
    'Decision_Tree': (DecisionTreeClassifier(random_state=42), {
        'max_depth': [15 ,18],  # Maximum depth of the tree
        'min_samples_split': [10, 12, 14],  # Minimum samples required to split a node
        'min_samples_leaf': [ 3]  # Minimum samples required at each leaf node
    }),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [3],  # Number of neighbors to consider for prediction
        'metric': [ 'manhattan']  # Distance metric
    }),
    'SGDClassifier': (SGDClassifier(random_state=42), {
        'loss': ['hinge'],  # Loss function
        'alpha': [0.01],  # Regularization parameter (penalty for model complexity)
        'learning_rate': [ 'optimal']  # Learning rate schedule
    }),
    'MLPclassifier': (MLPClassifier(random_state=42), {
        'hidden_layer_sizes': [ (100, 100)],  # Number of neurons in each hidden layer
        'activation': ['relu'],  # Activation function for hidden layers
        'alpha': [0.00001, 0.0001, 0.001],  # Regularization parameter (L1 penalty)
        'solver': ['adam'],  # Optimization algorithm
        'learning_rate_init': [0.0001]  ,# Initial learning rate
        'max_iter': [500]
    })
}

models_ex2 = {
    'Linear_Regression': (LinearRegression(), {}), # No hyperparameters to tune
    'Lasso_Regression': (Lasso(random_state=42), {'alpha': [0.0001]}),# L1 penalty (sparsity)
    'Ridge_Regression': (Ridge(random_state=42), {'alpha': [ 10]}),# L2 penalty (regularization)
    'SVR': (SVR(), {'C': [250], 'gamma': [0.1], 'kernel': ['rbf']}),# Kernel & complexity vs. error
    'Gradient_Boosting': (HistGradientBoostingRegressor(random_state=42), {'max_depth': [20], 'learning_rate': [0.1]}), # Tree depth & learning rate
    'Random_Forest_Regressor': (RandomForestRegressor(random_state=42), {'n_estimators': [3000], 'max_depth': [20]}),# Number of trees & depth
    'KNN_Regressor': (KNeighborsRegressor(), {'n_neighbors': [7], 'metric': ['minkowski']}), # Neighbors & distance
    'SGDRegressor': (SGDRegressor(random_state=42), {'alpha': [0.01], 'learning_rate': ['optimal']}),# Regularization & learning rate
    'XGBRegressor': (xgb.XGBRegressor(random_state=42), {
        'n_estimators': [600],  # Number of boosting stages
        'max_depth': [3],  # Maximum depth of individual trees
        'learning_rate': [0.1],  # Step size in each boosting step
        'gamma': [0],  # Minimum loss reduction required for a split
        'subsample': [0.8],  # Fraction of samples to use for training each booster
        'colsample_bytree': [0.8],  # Fraction of features to use for training each booster
    })
}
