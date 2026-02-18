# RandomMachine

Random ensemble learning library that combines LightGBM and CatBoost with random depth selection for improved model diversity.

## Installation

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

**Random Mixed:**
- `RandomRegressor` - Randomly switches between CatBoost and LightGBM
- `RandomClassifier` - Randomly switches between CatBoost and LightGBM

## Tutorials

Interactive Jupyter notebooks in the `/docs` folder:
- [Tutorial](docs/tutorial.ipynb) - Getting started guide with **performance comparison vs plain LightGBM/CatBoost**
- [Advanced Guide](docs/advanced_guide.ipynb) - Optimization and best practices

```bash
cd docs/
jupyter notebook tutorial.ipynb
```

The tutorial includes side-by-side comparisons showing whether RandomMachine actually outperforms baseline models.

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
