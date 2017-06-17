import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#############################
## Load training data
#############################

## Load main dataset
d_rf = pd.read_csv('source_data/rec_flag_2015_16.csv')

## Load MWRA historical data
d_mwra_base = pd.read_csv('source_data/mwra_base_recflag.csv')

## Load DCR beach historical data
d_dcr = pd.read_csv('source_data/muni_dcr_hist.csv')

## Load rainfall data
d_logan = pd.read_csv('source_data/logan_hourly.csv')

## Load streamflow data
d_d_aberjona = pd.read_table('source_data/01102500-aberjona-day.txt', comment='#').iloc[1:]
d_d_alewife = pd.read_table('source_data/01103025-alewife-day.txt', comment='#').iloc[1:]
d_h_alewife = pd.read_table('source_data/01103025-alewife-hour.txt', comment='#').iloc[1:]

#############################
## Transform data
#############################

db_dfs = {'rf':d_rf, 'mwra':d_mwra_base, 'dcr':d_dcr}
flow_dfs = {'d_aberjona':d_d_aberjona, 'd_alewife':d_d_alewife, 'h_alewife':d_h_alewife}

## Times
for name in db_dfs:
	db_dfs[name].DateHour = pd.to_datetime(db_dfs[name].DateHour)

for name in flow_dfs:
	flow_dfs[name].datetime = pd.to_datetime(flow_dfs[name].datetime)

d_logan.datetime = pd.to_datetime(d_logan.datetime)
d_logan.index = d_logan.datetime

## Categoricals
col_cat = [
	'LocationID',
	'FlagID',
	'Weather',
	'WaterType',
	'WaterBodyID',
	'LocationTypeID',
	'Qualifier',
	]

col_cat_dummies = []
for name in db_dfs:
	for col in col_cat:
		df_dum = pd.get_dummies(db_dfs[name][col], prefix=col, drop_first=1, dummy_na=1)
		db_dfs[name] = pd.concat([db_dfs[name], df_dum], axis=1)
		col_cat_dummies += df_dum.columns.tolist()

col_cat_dummies = np.unique(col_cat_dummies)

## Generate interpolated hourly streamflow for aberjona
flow_dfs['d_aberjona'].index = flow_dfs['d_aberjona'].datetime
hour_series = pd.date_range(np.min(flow_dfs['d_aberjona'].datetime), np.max(flow_dfs['d_aberjona'].datetime), freq=pd.Timedelta(1, 'h'))
flow_dfs['h_aberjona'] = flow_dfs['d_aberjona'].ix[hour_series]
for col in ['agency_cd','site_no']:
	flow_dfs['h_aberjona'][col] = flow_dfs['h_aberjona'][col].fillna(method='ffill')

flow_dfs['h_aberjona']['datetime'] = flow_dfs['h_aberjona'].index
flow_dfs['h_aberjona']['64138_00060_00003'] = flow_dfs['h_aberjona']['64138_00060_00003'].astype(float).interpolate()

## Calculate streamflow derivative
flow_dfs['h_alewife'].index = flow_dfs['h_alewife'].datetime

## From USGS file:
# Data provided for site 01103025
#            TS   parameter     Description
#        168619       00060     Discharge, cubic feet per second
#         66065       00065     Gage height, feet
#         66067       72255     Mean water velocity for discharge computation, feet per second
flow_dfs['h_alewife']['flow_deriv'] = np.nan
flow_dfs['h_alewife']['flow_deriv'].loc[1:] = flow_dfs['h_alewife'][u'168619_00060'].astype(float).values[1:] - flow_dfs['h_alewife']['168619_00060'].astype(float).values[:-1]

flow_dfs['h_aberjona']['flow_deriv'] = np.nan
flow_dfs['h_aberjona']['flow_deriv'].loc[1:] = flow_dfs['h_aberjona']['64138_00060_00003'].astype(float).values[1:] - flow_dfs['h_aberjona']['64138_00060_00003'].astype(float).values[:-1]


#############################
## Generate training and testing matrices
#############################

## Filter to CharactericID=='ECOLI' and Units=='MPN/100ml'
d_rf_ecoli = db_dfs['rf'][(db_dfs['rf']["CharacteristicID"]=='ECOLI') & (db_dfs['rf']['Units']=='MPN/100ml')]

## Join streamflow data and derivatives
d_rf_ecoli['flow_alewife_current'] = flow_dfs['h_alewife']['168619_00060'].ix[d_rf_ecoli.DateHour].values
d_rf_ecoli['flow_alewife_deriv'] = flow_dfs['h_alewife']['flow_deriv'].ix[d_rf_ecoli.DateHour].values
d_rf_ecoli['flow_aberjona_current'] = flow_dfs['h_aberjona']['64138_00060_00003'].ix[d_rf_ecoli.DateHour].values
d_rf_ecoli['flow_aberjona_deriv'] = flow_dfs['h_aberjona']['flow_deriv'].ix[d_rf_ecoli.DateHour].values

## Join rainfall data on different timescales
def sum_rain(timestamp, hours=48, rain_series = d_logan['prcp_in']):
	sel = (rain_series.index >= timestamp - pd.Timedelta(hours, 'h')) & (rain_series.index < timestamp)
	return rain_series[sel].sum()

precip_ts = [12,24,48,72,96,120]
for t in precip_ts:
	d_rf_ecoli['precip_'+str(t)] = d_rf_ecoli.DateHour.apply(lambda x: sum_rain(x, hours=t))

## Gather variables
IVs = [
	'flow_alewife_current', 'flow_alewife_deriv','flow_aberjona_current','flow_aberjona_deriv',
	] + list(col_cat_dummies) + ['precip_'+str(t) for t in precip_ts]
IVs = [iv for iv in IVs if iv in d_rf_ecoli] # some columns came from other dataframes - eliminate them
IVs = [iv for iv in IVs if iv.startswith('Qualifier')==0] # does not make causal sense to include the bacterial results qualifier
DV = 'ResultValue'

# Instantiate matrices
X = d_rf_ecoli.ix[:,IVs].astype(float).values
Y = (d_rf_ecoli[DV].values > 1260).astype(float) # Boating standard

## Check for nulls
#d_rf_ecoli.ix[:,IVs].astype(float).isnull().sum()

## Turns out some flow values are null.  Fill with mean
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
imp.fit(X)
Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
X_imp = imp.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
	train_test_split(X_imp, Y, test_size=0.2, random_state=101, stratify=Y)

baseline = 1-y_test.mean()


#############################
## Train decision tree model
#############################

from sklearn.model_selection import GridSearchCV

dt_parameters = {'max_depth':range(2,7), 'min_samples_split':[2,5], 'min_samples_leaf':[1,3], 'class_weight':[None,'balanced']}

from sklearn.tree import DecisionTreeClassifier

clf_dt = GridSearchCV(
			DecisionTreeClassifier(),
			dt_parameters, n_jobs=4, cv=7).fit(X_train, y_train)

print clf_dt.best_score_ 
print clf_dt.best_params_

print clf_dt.score(X_test, y_test)  

## Visualize best tree
import pydotplus
from sklearn import tree
dot_data = tree.export_graphviz(clf_dt.best_estimator_, feature_names=IVs, out_file=None, filled=True) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("test_best_tree.pdf") 

#############################
## Train GBM model
#############################

from sklearn.ensemble import GradientBoostingClassifier

## See https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
gbm_parameters = {
	'n_estimators':[70,100,150], 
	'learning_rate':[0.025,.05,.07], 
	'max_depth':[3,4,5], 
	'min_samples_leaf':[3,5], 
	'min_samples_split':[2,3]}

clf_gbm = GridSearchCV(
			GradientBoostingClassifier(),
			gbm_parameters, n_jobs=4, cv=7).fit(X_train, y_train)

print clf_gbm.best_score_ 
print clf_gbm.best_params_

print clf_gbm.score(X_test, y_test)  




#############################
## Train linear regression model
#############################

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

ss_X = StandardScaler().fit(X_train)

clf_lr = LogisticRegressionCV(Cs=10, n_jobs=4, cv=7).fit(ss_X.transform(X_train), y_train)

print np.mean(clf_lr.scores_[1], axis=1)
print clf_lr.C_

print pd.Series(index = IVs, data = clf_lr.coef_[0])

print clf_lr.score(ss_X.transform(X_test), y_test)  


## With interactions
from sklearn.preprocessing import PolynomialFeatures

pf_int = PolynomialFeatures(degree=2)
pf_int.fit(ss_X.transform(X_train))

clf_lri = LogisticRegressionCV(Cs=10, n_jobs=4, cv=7).fit(pf_int.transform(ss_X.transform(X_train)), y_train)

print np.mean(clf_lri.scores_[1], axis=1)
print clf_lri.C_

print pd.Series(index = pf_int.get_feature_names(), data = clf_lri.coef_[0])

print clf_lri.score(pf_int.transform(ss_X.transform(X_test)), y_test)  



#############################
## Summary plot
#############################

## Accuracy comparison
plt.figure()
model_scores = {'Baseline\n(Majority rule classifier)':baseline * 100, 
				'Logistic Regression':clf_lr.score(ss_X.transform(X_test), y_test) * 100, 
				'Logistic Regression\n(w/ interactions)':clf_lri.score(pf_int.transform(ss_X.transform(X_test)), y_test) * 100, 
				'Decision Tree':clf_dt.score(X_test, y_test) * 100, 
				'Gradient Boosting':clf_gbm.score(X_test, y_test) * 100 
				}
plt.barh(np.arange(len(model_scores)), model_scores.values())
plt.yticks(np.arange(len(model_scores)), model_scores.keys())
plt.gca().set_xlim(80,100)
plt.xlabel('Accuracy score (%)')
plt.title('Performance on held out testing set')
plt.savefig('test_model_accuracy_comparison.png', bbox_inches='tight')

## AUC comparison
from sklearn.metrics import roc_curve, auc

def auc_calc(model, ytest=y_test, xtest=X_test):
	fpr, tpr, thresholds = roc_curve(ytest, model.predict_proba(xtest)[:, 1])
	return auc(fpr, tpr), fpr, tpr

model_scores_auc = {'Baseline\n(Majority rule classifier)':.5, 
				'Logistic Regression':auc_calc(clf_lr, xtest=ss_X.transform(X_test))[0], 
				'Logistic Regression\n(w/ interactions)':auc_calc(clf_lri, xtest=pf_int.transform(ss_X.transform(X_test)))[0], 
				'Decision Tree':auc_calc(clf_dt)[0], 
				'Gradient Boosting':auc_calc(clf_gbm)[0] 
				}
plt.figure()
plt.barh(np.arange(len(model_scores_auc)), model_scores_auc.values())
plt.yticks(np.arange(len(model_scores_auc)), model_scores_auc.keys())
plt.gca().set_xlim(.5, 1)
plt.xlabel('AUC score')
plt.title('Performance on held out testing set')
plt.savefig('test_model_auc_comparison.png', bbox_inches='tight')


## ROC curve
model_scores_pr = {'Baseline\n(Majority rule classifier)':([0,1],[0,1]), 
				'Logistic Regression':auc_calc(clf_lr)[1:], 
				'Logistic Regression\n(w/ interactions)':auc_calc(clf_lri, xtest=pf_int.transform(ss_X.transform(X_test)))[1:], 
				'Decision Tree':auc_calc(clf_dt)[1:], 
				'Gradient Boosting':auc_calc(clf_gbm)[1:] 
				}
plt.figure()
for model in model_scores_pr:
	plt.plot(*model_scores_pr[model], label=model)

plt.axis([0,1,0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Performance on held out testing set')
plt.legend()
plt.savefig('test_model_roc_comparison.png', bbox_inches='tight')


#############################
## Questions
#############################

"""
1. What factors are available at prediction time?  Are any other measurements like turbidity or pH available?
2. What are the LocationTypeID levels?
3. Is it ok to focus on predicting ECOLI measurements only?
4. I get an error at this link: https://www.dropbox.com/s/akbcbtynztuehbi/myrwa-recflag-2015.html?dl=0
"""

#############################
## Thoughts
#############################

"""
* Should optimize for recall - want to play it safe
"""

