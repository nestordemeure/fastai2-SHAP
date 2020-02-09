# Fast Shap
# Bindings to use the Shap interpretability library (https://github.com/slundberg/shap) with fastai2's tabular learner
# source: https://github.com/nestordemeure/fastai2-SHAP/blob/master/fastShap.py
import random
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

def _get_class_info(class_id, class_names):
    "returns the class name associated with an index or vice versa"
    if type(class_id) == int:
        class_index = class_id
        class_name = class_names[class_index]
    else:
        class_name = class_id
        class_index = class_names.o2i[class_name]
    return (class_name, class_index)

def _get_values(interpreter, class_id=0):
    "Gets `shap_value` and `expected_value`, dealing with the possibility that the model is multi-output."
    shap_values = interpreter.shap_values
    expected_value = interpreter.explainer.expected_value
    if interpreter.is_multi_output:
        (class_name, class_index) = _get_class_info(class_id, interpreter.class_names) 
        print("Classification model detected, we will display the score for the class", class_name)
        print("(use `class_id` to specify another class)")
        shap_values = shap_values[class_index]
        expected_value = expected_value[class_index]
    return (shap_values, expected_value)

class ShapInterpretation():
    "Encapsulate bindings with the Shap interpretation library."
    def __init__(self, learn:TabularLearner, test_data:pd.DataFrame=None, link="identity", nsamples="auto", l1_reg="auto", **kwargs):
        """
        Uses Shap value to interpret the output of a learner for some test data
        
        test_data : None or a pandas dataframe
            The data for which the shap values will be computed.
            By default, 100 random rows of the train data will be used.
        
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
        # extracts model and data from the learner
        self.model = learn.model
        self.dls = learn.dls
        self.class_names = learn.dl.vocab
        # create an explainer for the model
        train_data = learn.dls.all_cols
        predict_function = partial(_predict, model=learn.model, dls=learn.dls)
        self.explainer = shap.SamplingExplainer(predict_function, train_data, **kwargs)
        #self.explainer = shap.KernelExplainer(predict_function, train_data, **kwargs) # use only for small dataset or sample
        # computes shap values for the test data
        self.test_data = train_data.sample(n=min(100, len(train_data)),replace=False) if test_data is None else learn.dls.test_dl(test_data).all_cols
        self.shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples, l1_reg=l1_reg)
        # flags used to indure the proper working of the library
        self.is_multi_output = type(self.shap_values) == list
            
    def summary_plot(self, **kwargs):
        """
        Displays the Shap values (which can be interpreted as feature importance).
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/summary.py
        """
        return shap.summary_plot(self.shap_values, self.test_data, class_names=self.class_names, **kwargs)

    def dependence_plot(self, variable_name:str, class_id=0, **kwargs):
        """
        Plots the value of the variable on the x-axis and the SHAP value of the same variable on the y-axis.
        This shows how the model depends on the given variable. 
        Vertical dispersion of the data points represents interaction effects.
        Grey ticks along the y-axis are data points where the variable's value was NaN.
        
        `class_id` is used to indicate the class of interest for classification models, it can ba an int or a string
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/dependence.py
        For more informations, see: https://github.com/slundberg/shap/blob/master/notebooks/plots/dependence_plot.ipynb
        """
        shap_values, expected_value = _get_values(self, class_id)
        # TODO why does the graph come out flat ? (overlap between train and test ?)
        return shap.dependence_plot(variable_name, shap_values, self.test_data, **kwargs)

    def waterfall_plot(self, row_index=None, class_id=0, **kwargs):
        """
        Plots an explanation of a single prediction as a waterfall plot.
        
        `row_index` is the index of the row in `test_data` that will be analyzed, if it is None it will be drawed at random
        `class_id` is used to indicate the class of interest for classification models, it can ba an int or a string
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/waterfall.py
        """
        shap_values, expected_value = _get_values(self, class_id)
        nb_rows = shap_values.shape[0]
        row_index = random.randint(0,nb_rows-1) if row_index is None else row_index
        print("Displaying row", row_index, "of", nb_rows, "(use `row_index` to specify another row)")
        feature_names = self.test_data.columns
        return shap.waterfall_plot(expected_value, shap_values[row_index,:], feature_names=feature_names, **kwargs)

    def force_plot(self, class_id=0, matplotlib=False, **kwargs):
        """
        Visualize the SHAP values with an additive force layout.
        
        `class_id` is used to indicate the class of interest for classification models, it can ba an int or a string
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/force.py
        """
        shap_values, expected_value = _get_values(self, class_id)
        if not matplotlib: shap.initjs()
        return shap.force_plot(expected_value, shap_values, self.test_data, matplotlib=matplotlib, **kwargs)

    def decision_plot(self, class_id=0, **kwargs):
        """
        Visualize model decisions using cumulative SHAP values. Each colored line in the plot represents the model
        prediction for a single observation. Note that plotting too many samples at once can make the plot unintelligible.
        
        `class_id` is used to indicate the class of interest for classification models, it can ba an int or a string
        
        For an up-to-date list of the parameters, see: https://github.com/slundberg/shap/blob/master/shap/plots/decision.py
        For more informations, see: https://github.com/slundberg/shap/blob/master/notebooks/plots/decision_plot.ipynb
        """
        # NOTE: there is a shap.multioutput_decision_plot but it uses a single row
        shap_values, expected_value = _get_values(self, class_id)
        return shap.decision_plot(expected_value, shap_values, self.test_data, **kwargs)

# TODO: add method to build directly from learner
# learn.interpret_shap()
# why is dependence plot flat ?