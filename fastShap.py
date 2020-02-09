
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

def _one_class_shap_value(shap_values, class_index=0):
    "Takes shap values and get the values for a single output class in the case of a multi-output model."
    if type(shap_values) == list:
        print("Multi-output model detected, we will display the information for the class", class_index)
        print("use `class_index` to specify another class.")
        return shap_values[class_index]
    else:
        return shap_values

# https://github.com/fastai/fastai2/blob/79a9ba75864350b9d9e4278e640c8d285a805077/nbs/20_interpret.ipynb
class ShapInterpretation():
    "Used to encapsulate bindings with the Shap interpretation library"
    def __init__(self, learn:TabularLearner, test_data:pd.DataFrame):
        "Uses Shap value to interpret the output of a learner for some test data"
        # TODO use sample of train data as defalt test data
        # extracts model and data from the learner
        self.model = learn.model
        self.dls = learn.dls
        # create an explainer for the test data
        train_data = learn.dls.all_cols
        predict_function = partial(_predict, model=learn.model, dls=learn.dls)
        self.explainer = shap.SamplingExplainer(predict_function, train_data)
        #self.explainer = shap.KernelExplainer(self.pred, train_data)
        self.test_data = learn.dls.test_dl(test_data).all_cols
        self.shap_values = self.explainer.shap_values(self.test_data, l1_reg=False) # TODO why no L1 reg ?
        # flags used to indure the proper working of the library
        self._js_initialized = False
        #self.is_multi_output = type(self.shap_values) == list

    def initjs(self):
        "inits the javascript used by Shap for its plots if it has not been done already"
        if not self._js_initialized:
            shap.initjs()
            self._js_initialized = True
            
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
        """
        shap_values = _one_class_shap_value(self.shap_values, class_index)
        # TODO why does the graph come out flat ? (overlap between train and test ?)
        return shap.dependence_plot(variable_name, shap_values, self.test_data, **kwargs)

    def waterfall_plot(self, **kwargs):
        self.initjs()
        # TODO shap.waterfall_plot(*args, **kwargs)

    def force_plot(self, class_index=0, matplotlib=False, **kwargs):
        """
        Visualize the SHAP values with an additive force layout.
        
        `class_index` is used if the model has several ouputs.
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/force.py
        """
        if type(self.shap_values) == list:
            print("Multi-output model detected, we will display the information for the class", class_index)
            print("use the `class_index` parameter to specify another class.")
            shap_values = self.shap_values[class_index]
            expected_value = self.explainer.expected_value[class_index]
        else:
            shap_values = self.shap_values
            expected_value = self.explainer.expected_value
        if not matplotlib: self.initjs()
        return shap.force_plot(expected_value, shap_values, self.test_data, matplotlib=matplotlib, **kwargs)

    def image_plot(self, **kwargs):
        self.initjs()
        # TODO shap.image_plot(*args, **kwargs)

    def decision_plot(self, **kwargs):
        self.initjs()
        # TODO shap.decision_plot(*args, **kwargs)

# learn.interpret_shap()

# https://shap.readthedocs.io/en/latest/