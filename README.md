# RandomMachine

[![PyPI Version](https://img.shields.io/pypi/v/randommachine.svg)](https://pypi.org/project/randommachine/) [![License](https://img.shields.io/pypi/l/randommachine.svg)](https://github.com/ghiffaryr/randommachine/blob/master/LICENSE.md)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18687466.svg)](https://doi.org/10.5281/zenodo.18687466)

Random ensemble learning library that extends gradient boosting by randomly sampling base learners from a pool of LightGBM, CatBoost, XGBoost, and arbitrary scikit-learn-compatible estimators for improved ensemble diversity.

## Installation

```bash
pip install randommachine
```

Or install from source:

```bash
git clone https://github.com/ghiffaryr/randommachine.git
cd randommachine
pip install -e .
```

## Quick Start

### Regression

```python
from randommachine import RandomLGBMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomLGBMRegressor(num_iterations=20, learning_rate=0.5, random_state=42)
model.fit(X_train, y_train, X_eval=X_test, y_eval=y_test)

predictions = model.predict(X_test)
```

### Classification

```python
from randommachine import RandomCatBoostClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomCatBoostClassifier(num_iterations=20, learning_rate=0.5)
model.fit(X_train, y_train, X_eval=X_test, y_eval=y_test)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Available Models

**LightGBM-based:**
- `RandomLGBMRegressor` - Regression with random LightGBM base learners
- `RandomLGBMClassifier` - Classification with random LightGBM base learners

**CatBoost-based:**
- `RandomCatBoostRegressor` - Regression with random CatBoost base learners
- `RandomCatBoostClassifier` - Classification with random CatBoost base learners

**XGBoost-based:**
- `RandomXGBRegressor` - Regression with random XGBoost base learners
- `RandomXGBClassifier` - Classification with random XGBoost base learners

**Generic (user-defined pool):**
- `RandomRegressor` - Mix any sklearn-compatible regressors with custom probabilities
- `RandomClassifier` - Mix any sklearn-compatible classifiers with custom probabilities

## Tutorial

An interactive Jupyter notebook is available in the `/docs` folder:
- [Tutorial](docs/tutorial.ipynb) - Getting started guide with **performance comparison vs plain LightGBM, CatBoost, and XGBoost baselines**

```bash
cd docs/
jupyter notebook tutorial.ipynb
```

The tutorial includes side-by-side comparisons showing RandomMachine's improvement over fixed-family baselines.

## Development

Run tests:
```bash
make test          # Run all tests
make test-cov      # With coverage report
```

Format code:
```bash
make format        # Black formatting
make lint          # Flake8 linting
```

## License

MIT License - see [LICENSE](LICENSE)
