import pandas as pd
from IPython.display import display
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline


class FitPredictGo:
    def __init__(self, name, scoring, random_state=1206):
        self.name = name
        self.scoring = scoring
        self.simple_fitted = None
        self.fitted = {}
        self.gs_fitted = None
        self.rs_fitted = None
        self.fit_type = None
        self.random_state = random_state

    def fit(self, preprocessors, model, X_train, y_train):
        self.fit_type = "simple fit"
        self.set_pipeline(preprocessors, model)
        self.fitted[self.fit_type] = self.pipe.fit(X_train, y_train)
        return "Fitted"

    def fit_cv(self, preprocessors, model, X_train, y_train, cv_folds, results=True):
        self.set_pipeline(preprocessors, model)
        self.fit_type = "cv"
        cv = cross_validate(
            self.pipe, X_train, y_train, cv=cv_folds, scoring=self.scoring
        )
        self.fitted[self.fit_type] = self.pipe.fit(X_train, y_train)
        if results:
            return self.get_results(cv)
        return cv

    def fit_search(
        self,
        preprocessors,
        model,
        param_grid,
        search_type,
        X_train,
        y_train,
        cv_folds,
        results=True,
    ):
        self.set_pipeline(preprocessors, model)
        self.fit_type = search_type
        if search_type == "Grid":
            search = GridSearchCV(
                self.pipe, param_grid, n_jobs=-1, cv=cv_folds, scoring=self.scoring
            )
        else:
            search = RandomizedSearchCV(
                self.pipe,
                param_grid,
                n_jobs=1,
                cv=cv_folds,
                scoring=self.scoring,
                random_state=self.random_state,
            )
        search.fit(X_train, y_train)
        self.fitted[self.fit_type] = search.best_estimator_
        if results:
            return self.get_results(search)
        return search

    def predict(self, X_test, predict_type):
        return self.fitted[predict_type].predict(X_test)

    def predict_proba(self, X_test, predict_type):
        return self.fitted[predict_type].predict_proba(X_test)[:, 1]

    def set_pipeline(self, preprocessors, model):
        estimators = preprocessors + [model]
        self.pipe = make_pipeline(*estimators)
        display(self.pipe)

    def get_results(self, results):
        if self.fit_type == "cv":
            result = (
                self.name,
                round(results["fit_time"].mean(), 3),
                round(results["score_time"].mean(), 3),
                round(results["test_score"].mean(), 3),
                round(results["test_score"].std(), 3),
            )
            self.fit_type = None
        else:
            res = results.cv_results_
            idx = results.best_index_
            result = (
                f"{self.fit_type} enhanced {self.name}",
                round(res["mean_fit_time"][idx], 3),
                round(res["mean_score_time"][idx], 3),
                round(res["mean_test_score"][idx], 3),
                round(res["std_test_score"][idx], 3),
            )
            self.fit_type = None
        return result

    def get_best_estimator(self, predict_type):
        return self.fitted[predict_type]


class Results:
    def __init__(self):
        self.df = pd.DataFrame(
            [],
            columns=[
                "Наименование модели",
                "Время обучения, сек.",
                "Время предсказания, сек.",
                "Среднее значение метрики при CV",
                "СКО метрики при CV",
            ],
        )

    def update(self, new_row, show=False):
        del_row = new_row[0]
        self.df = self.df[~(self.df["Наименование модели"] == del_row)]
        try:
            rows_exist = max(self.df.index)
        except ValueError:
            rows_exist = -1
        finally:
            self.df.loc[rows_exist + 1] = new_row
        if show:
            print(
                self.df.sort_values(
                    by="Среднее значение метрики при CV", ascending=False
                )
            )


def prepare_split(df, goal, columns, val=False, test_valid_size=0.2, random_state=1206):
    X = df[columns]
    y = df[goal]

    if not val:
        # Разобьем данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_valid_size, random_state=random_state
        )
        return X_train, y_train, X_test, y_test
    else:
        val = test_valid_size
        # Разобьем данные на обучающую, тестовую и валидационную выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_valid_size * 0.5, random_state=random_state
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            stratify=y_train,
            test_size=test_valid_size * 0.5,
            random_state=random_state,
        )
        return (
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
        )
