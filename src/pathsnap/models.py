"""Model zoo. Every model is a sklearn Pipeline so imputation/scaling is fitted inside each fold.

`max_train_rows` subsamples training rows (by random rows, seeded) for slow models; this is
recorded in the experiment log and must be stated in the report.
"""
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _impute():
    return SimpleImputer(strategy="median", add_indicator=True)


def get_model(name, seed=42, **params):
    """Returns (pipeline, family, max_train_rows)."""
    if name == "logreg":
        p = dict(C=1.0, max_iter=500) | params
        return make_pipeline(_impute(), StandardScaler(), LogisticRegression(**p)), "Linear", None
    if name == "gaussian_nb":
        return make_pipeline(_impute(), StandardScaler(), GaussianNB(**params)), "Probabilistic", None
    if name == "knn":
        p = dict(n_neighbors=25, weights="distance", n_jobs=-1) | params
        return make_pipeline(_impute(), StandardScaler(), KNeighborsClassifier(**p)), "Instance-based", 150_000
    if name == "random_forest":
        p = dict(n_estimators=150, max_depth=18, min_samples_leaf=5, max_samples=0.3, n_jobs=-1, random_state=seed) | params
        return make_pipeline(_impute(), RandomForestClassifier(**p)), "Ensemble (bagging)", None
    if name == "extra_trees":
        p = dict(n_estimators=150, max_depth=18, min_samples_leaf=5, max_samples=0.3, bootstrap=True, n_jobs=-1, random_state=seed) | params
        return make_pipeline(_impute(), ExtraTreesClassifier(**p)), "Ensemble (randomised trees)", None
    if name == "hist_gb":
        p = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, random_state=seed) | params
        return HistGradientBoostingClassifier(**p), "Boosting", None   # handles NaN natively
    if name == "mlp":
        p = dict(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, max_iter=40,
                 early_stopping=True, n_iter_no_change=5, random_state=seed) | params
        return make_pipeline(_impute(), StandardScaler(), MLPClassifier(**p)), "Neural network", 400_000
    raise ValueError(name)


ALL_MODELS = ["logreg", "gaussian_nb", "knn", "random_forest", "extra_trees", "hist_gb", "mlp"]
