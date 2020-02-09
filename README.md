# Fast-SHAP

This repository includes bindings to use the [SHAP](https://github.com/slundberg/shap) interpretability library with [fastai2](http://dev.fast.ai/)'s tabular learner.

## Usage

To use this code, you will first need to install [SHAP](https://github.com/slundberg/shap) : `pip install shap`

The interpretation is done via the `ShapInterpretation` class (see the [Demo2 notebook](https://github.com/nestordemeure/fastai2-SHAP/blob/master/Demo2.ipynb) for an example):

```python
# model
learn = tabular_learner(dataloaders, layers=[200,100], metrics=accuracy)
learn.fit(1, 1e-2)

# interpretation
interpreter = ShapInterpretation(learn)
interpreter.summary_plot()
```

## TODO

Why is `dependence_plot` flat ?

Add the possibility to use `waterfall_plot` with a given row that was not in `test_data`.