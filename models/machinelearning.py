from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor


def get_prediction_type(model_id,type=None):
    classification = [0,1,6,11,2,3,5,61,7]
    regression = [1,6,12,3,5,62,7]
    clustering = [4]
    if model_id in classification and model_id in regression :
        if model_id in [1,6] and type == 1 : return ["Classification"]
        if model_id in [1,6] and type == 2 : return ["Regression"]
        return ["Classification","Regression"]
    elif model_id in classification:
        return ["Classification"]
    elif model_id in regression:
        return ["Regression"]
    elif model_id in clustering:
        return ["Clustering"]


def get_model(model_id,prediction_type,**kwarg):
    model = None
    if model_id == 0:
        model = LogisticRegression(**kwarg)
    if model_id == 1:        
        if prediction_type == "Classification":model = DecisionTreeClassifier(**kwarg)
        else : model = DecisionTreeRegressor(**kwarg)
    if model_id == 2:        
        model = GaussianNB()
    if model_id == 3:        
        if prediction_type == "Classification":model = SVC(**kwarg)
        else : model = SVR(**kwarg)
    if model_id == 4:        
        model = KMeans(**kwarg)
    if model_id == 5:        
        if prediction_type == "Classification":model = KNeighborsClassifier(**kwarg)
        else : model = KNeighborsRegressor(**kwarg)
    if model_id == 6:        
        if prediction_type == "Classification":model = RandomForestClassifier(**kwarg)
        else : model = RandomForestRegressor(**kwarg)
    if model_id == 7:
        if prediction_type == "Classification":model = MLPClassifier()
        else : model = MLPRegressor()
    return model