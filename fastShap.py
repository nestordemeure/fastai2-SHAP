
# Fast Shap
# Bindings to use the Shap interpretability library (https://github.com/slundberg/shap) with fastai2's tabular learner
# source: TODO
import shap
from fastai2.tabular.all import *

__all__ = ['ShapInterpretation']

def _predict(data, model, dls):
    "Takes data, as shap produces it, and returns a prediction using the model."
    # gets the data into torch
    device = dls.device
    nb_cat_cols = len(dls.train_ds.cat_names)
    nb_cont_cols = len(dls.train_ds.cont_names)
    x_cat = torch.from_numpy(data[:, :nb_cat_cols]).to(device, torch.int64)
    x_cont = torch.from_numpy(data[:, -nb_cont_cols:]).to(device, torch.float32)
    # computes a prediction with the model
    pred_proba = model(x_cat, x_cont).detach().to('cpu').numpy()
    return pred_proba

def _get_values(interpreter, class_index:int=0):
    "Gets `shap_value` and `expected_value`, dealing with the possibility that the model is multi-output."
    shap_values = interpreter.shap_values
    expected_value = interpreter.explainer.expected_value
    if interpreter.is_multi_output:
        print("Multi-output model detected, we will display the information for the class", class_index, "of", len(interpreter.shap_values))
        print("use the `class_index` parameter to specify another class.")
        shap_values = shap_values[class_index]
        expected_value = expected_value[class_index]
    return (shap_values, expected_value)

# https://github.com/fastai/fastai2/blob/79a9ba75864350b9d9e4278e640c8d285a805077/nbs/20_interpret.ipynb
class ShapInterpretation():
    "Used to encapsulate bindings with the Shap interpretation library"
    def __init__(self, learn:TabularLearner, test_data:pd.DataFrame, link="identity", nsamples="auto", l1_reg="auto", **kwargs):
        """
        Uses Shap value to interpret the output of a learner for some test data
        
        link : "identity" or "logit"
            A generalized linear model link to connect the feature importance values to the model
            output. Since the feature importance values, phi, sum up to the model output, it often makes
            sense to connect them to the ouput with a link function where link(outout) = sum(phi).
            If the model output is a probability then the LogitLink link function makes the feature
            importance values have log-odds units.
        
        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction.
            More samples lead to lower variance estimates of the SHAP values.
            
        l1_reg : "num_features(int)", "auto", "aic", "bic", or float
            The l1 regularization to use for feature selection (the estimation procedure is based on
            a debiased lasso). The auto option currently uses "aic" when less that 20% of the possible sample
            space is enumerated, otherwise it uses no regularization.
        """
        # TODO use sample of train data as defalt test data
        
        # extracts model and data from the learner
        self.model = learn.model
        self.dls = learn.dls
        # create an explainer for the model
        train_data = learn.dls.all_cols
        predict_function = partial(_predict, model=learn.model, dls=learn.dls)
        self.explainer = shap.SamplingExplainer(predict_function, train_data, **kwargs)
        #self.explainer = shap.KernelExplainer(self.pred, train_data, **kwargs)
        # computes shap values for the test data
        self.test_data = learn.dls.test_dl(test_data).all_cols
        self.shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples, l1_reg=l1_reg)
        # flags used to indure the proper working of the library
        self.is_multi_output = type(self.shap_values) == list
            
    def summary_plot(self, **kwargs):
        """
        Displays the Shap values (which can be interpreted as feature importance).
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/summary.py
        """
        return shap.summary_plot(self.shap_values, self.test_data, **kwargs)

    def dependence_plot(self, variable_name:str, class_index=0, **kwargs):
        """
        Plots the value of the variable on the x-axis and the SHAP value of the same variable on the y-axis.
        This shows how the model depends on the given variable. 
        Vertical dispersion of the data points represents interaction effects.
        Grey ticks along the y-axis are data points where the variable's value was NaN.
        
        `class_index` is used if the model has several ouputs.
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/dependence.py
        For more informations, see: https://github.com/slundberg/shap/blob/master/notebooks/plots/dependence_plot.ipynb
        """
        shap_values, expected_value = _get_values(self, class_index)
        # TODO why does the graph come out flat ? (overlap between train and test ?)
        return shap.dependence_plot(variable_name, shap_values, self.test_data, **kwargs)

    def waterfall_plot(self, row_index=0, class_index=0, **kwargs):
        """
        Plots an explantion of a single prediction as a waterfall plot.
        
        `row_index` is the index of the row in `test_data` that will be analyzed
        `class_index` is used if the model has several ouputs.
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/waterfall.py
        """
        shap_values, expected_value = _get_values(self, class_index)
        print("displaying row", row_index, "of", shap_values.shape[0])
        feature_names = self.test_data.columns
        shap.waterfall_plot(expected_value, shap_values[row_index,:], feature_names=feature_names, **kwargs)

    def force_plot(self, class_index=0, matplotlib=False, **kwargs):
        """
        Visualize the SHAP values with an additive force layout.
        
        `class_index` is used if the model has several ouputs.
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/force.py
        """
        shap_values, expected_value = _get_values(self, class_index)
        if not matplotlib: shap.initjs()
        return shap.force_plot(expected_value, shap_values, self.test_data, matplotlib=matplotlib, **kwargs)

    def decision_plot(self, class_index=0, **kwargs):
        """
        Visualize model decisions using cumulative SHAP values. Each colored line in the plot represents the model
        prediction for a single observation. Note that plotting too many samples at once can make the plot unintelligible.
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/decision.py
        For more informations, see: https://github.com/slundberg/shap/blob/master/notebooks/plots/decision_plot.ipynb
        """
        # NOTE there is a shap.multioutput_decision_plot but it expects expected_value to be a list wich it isn't 
        # (bug in shap?)
        # shap.multioutput_decision_plot(self.explainer.expected_value, self.shap_values, row_index, **kwargs)
        shap_values, expected_value = _get_values(self, class_index)
        shap.decision_plot(expected_value, shap_values, self.test_data, **kwargs)

# learn.interpret_shap()