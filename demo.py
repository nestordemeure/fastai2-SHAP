from fastai2.tabular.all import *
from fastShap import *

# dataset
path = untar_data(URLs.ADULT_SAMPLE)
dataframe = pd.read_csv(path/'adult.csv')
# variables
dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
# preprocessing
procs = [Categorify, FillMissing, Normalize]
splits = IndexSplitter(list(range(800,1000)))(range_of(dataframe))
to = TabularPandas(dataframe, procs, cat_names, cont_names, y_names="salary", splits=splits)
# assemble data loader
dataloaders = to.dataloaders()

# model
learn = tabular_learner(dataloaders, layers=[200,100], metrics=accuracy)
learn.fit(1, 1e-2)

# interpretation
test_data = dataframe.iloc[:100] # dataframe containing the 100 first rows
interpreter = ShapInterpretation(learn, test_data)

interpreter.force_plot()
interpreter.summary_plot()
interpreter.dependence_plot('age')
