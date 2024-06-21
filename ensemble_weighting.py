# get weights for ensemble

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def get_weights(preprocessor,data,target):

    data_train, data_test, target_train, target_test = train_test_split(data,
                                                                        target, random_state=42,
                                                                        test_size=.30)

    #balance train data
    data_oversampled, target_oversampled = resample(data_train[target == 1],
                                        target_train[target== 1],
                                        replace=True,
                                        n_samples=target_train[target_train == 0].shape[0],
                                        random_state=123)

    data_balanced = pd.concat((data_train[target_train == 0], data_oversampled))
    target_balanced = pd.concat((target_train[target_train == 0], target_oversampled))

    f_score = []
    for model in [RandomForestClassifier(n_estimators=500), HistGradientBoostingClassifier(max_iter=500),
                  LogisticRegression(max_iter=500), AdaBoostClassifier(n_estimators=500), SVC()]:
        model = make_pipeline(preprocessor, model)
        model.fit(data_balanced, target_balanced)
        target_pred = model.predict(data_test)
        p, r, f, _ = precision_recall_fscore_support(target_test, target_pred, average='binary')
        f_score.append(f)
    return(f_score)