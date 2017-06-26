import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.ion()

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

## Lookup tables
l_locationtype = pd.read_table('source_data/LocationTypeIDs.tsv', sep='\t')
l_locationtype.index = l_locationtype['ID']

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

## LocationTypeID names
for name in db_dfs:
	db_dfs[name]['LocationTypeID'] = db_dfs[name]['LocationTypeID'].apply(lambda x: l_locationtype['LocationTypeName'].ix[x]).values

## Categoricals
col_cat = [
	'LocationID',
	#'FlagID',
	#'Weather',
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
d_rf_bact = db_dfs['rf'][\
				((db_dfs['rf']["CharacteristicID"]=='ECOLI') & (db_dfs['rf']['Units']=='MPN/100ml')) |\
				((db_dfs['rf']["CharacteristicID"]=='ENT') & (db_dfs['rf']['Units']=='MPN/100ml'))\
				  ]

## Filter out flagged rows
d_rf_bact = d_rf_bact[d_rf_bact['FlagID'].isnull()]

## Join streamflow data and derivatives
d_rf_bact['flow_alewife_current'] = flow_dfs['h_alewife']['168619_00060'].ix[d_rf_bact.DateHour].values
d_rf_bact['flow_alewife_deriv'] = flow_dfs['h_alewife']['flow_deriv'].ix[d_rf_bact.DateHour].values
d_rf_bact['flow_aberjona_current'] = flow_dfs['h_aberjona']['64138_00060_00003'].ix[d_rf_bact.DateHour].values
d_rf_bact['flow_aberjona_deriv'] = flow_dfs['h_aberjona']['flow_deriv'].ix[d_rf_bact.DateHour].values

## Join rainfall data on different timescales
def sum_rain(timestamp, hours=[36,48], rain_series = d_logan['prcp_in']):
	sel = (rain_series.index >= timestamp - pd.Timedelta(hours[1], 'h')) & (rain_series.index < timestamp - pd.Timedelta(hours[0], 'h'))
	return rain_series[sel].sum()

precip_ts = [0,12,24,48,72,96,120]
for i in range(len(precip_ts)-1):
	d_rf_bact['precip_'+str(precip_ts[i])+'_'+str(precip_ts[i+1])] = d_rf_bact.DateHour.apply(lambda x: sum_rain(x, hours=[precip_ts[i], precip_ts[i+1]]))

## Gather variables
IVs = [
	'flow_alewife_current', 'flow_alewife_deriv','flow_aberjona_current','flow_aberjona_deriv',
	] + list(col_cat_dummies) + ['precip_'+str(precip_ts[i])+'_'+str(precip_ts[i+1]) for i in range(len(precip_ts)-1)]
DV = 'ResultValue'
IVs = [iv for iv in IVs if iv in d_rf_bact] # some columns came from other dataframes - eliminate them
IVs = [iv for iv in IVs if iv.startswith('Qualifier')==0] # does not make causal sense to include the bacterial results qualifier


## Generate flow and rainfall data for all of 2016 for testing purposes
hrs_2016 = pd.date_range(pd.Timestamp('2016-01-01 00:00:00'), pd.Timestamp('2016-12-31 00:00:00'), freq=pd.Timedelta(4, 'h'))
df_2016 = pd.DataFrame(data = {'DateHour':hrs_2016.values})
## Load hourly flow data
df_2016['flow_alewife_current'] = flow_dfs['h_alewife']['168619_00060'].ix[hrs_2016].values
df_2016['flow_alewife_deriv'] = flow_dfs['h_alewife']['flow_deriv'].ix[hrs_2016].values
## Missing some values, so interpolate
df_2016['flow_alewife_current'].interpolate('linear', inplace=True)
df_2016['flow_alewife_current'].fillna(0, inplace=True) # still end up with some nans after interpolation - not sure why, but should be ok to fill with 0s for testing purposes
df_2016['flow_alewife_deriv'].interpolate('linear', inplace=True)
df_2016['flow_aberjona_current'] = flow_dfs['h_aberjona']['64138_00060_00003'].ix[hrs_2016].values
df_2016['flow_aberjona_deriv'] = flow_dfs['h_aberjona']['flow_deriv'].ix[hrs_2016].values
## Calculate hourly precipitation
for i in range(len(precip_ts)-1):
	df_2016['precip_'+str(precip_ts[i])+'_'+str(precip_ts[i+1])] = \
		df_2016['DateHour'].apply(lambda x: sum_rain(x, hours=[precip_ts[i], precip_ts[i+1]]))


standard_limits = {
	'boating':{'ECOLI':1260,'ENT':350},
	'swimming':{'ECOLI':235,'ENT':104},
	}
for standard in standard_limits:

	# Instantiate matrices
	X = d_rf_bact.ix[:,IVs].astype(float).values
	# Boating standard
	Y = ((d_rf_bact[DV].values > standard_limits[standard]['ECOLI']) & (d_rf_bact['CharacteristicID'] == 'ECOLI')).astype(float) +\
		((d_rf_bact[DV].values > standard_limits[standard]['ENT']) & (d_rf_bact['CharacteristicID'] == 'ENT')).astype(float)

	## Check for nulls
	#d_rf_bact.ix[:,IVs].astype(float).isnull().sum()

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
	graph.write_pdf('test_'+standard+'_best_tree.pdf') 

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
	
	IVs_int = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(IVs,p) for p in pf_int.powers_]]

	#from patsy import dmatrices
	#y, X = dmatrices('depvar ~ C(var1)*C(var2)', df, return_type="dataframe")

	clf_lri = LogisticRegressionCV(Cs=10, n_jobs=4, cv=7).fit(pf_int.transform(ss_X.transform(X_train)), y_train)

	print np.mean(clf_lri.scores_[1], axis=1)
	print clf_lri.C_

	print pd.Series(index = pf_int.get_feature_names(), data = clf_lri.coef_[0])

	print clf_lri.score(pf_int.transform(ss_X.transform(X_test)), y_test)  


	#############################
	## Sample predictions
	#############################

	## Model functions
	model_pred_f = {'Baseline\n(Majority rule classifier)':lambda x: np.zeros(len(x)), 
					'Logistic Regression':lambda x: clf_lr.predict_proba(ss_X.transform(x))[:,1], 
					'Logistic Regression\n(w/ interactions)':lambda x: clf_lri.predict_proba(pf_int.transform(ss_X.transform(x)))[:,1], 
					'Decision Tree':lambda x: clf_dt.predict_proba(x)[:,1], 
					'Gradient Boosting':lambda x: clf_gbm.predict_proba(x)[:,1] 
					}
	
	## Input vector using 2016 rainfall and flow data at each location
	pred_X_loc = []
	# Step through locations
	for loc in d_rf_bact.LocationID.unique():
		df = pd.DataFrame(np.zeros([len(hrs_2016), len(IVs)]), columns = IVs)
		if 'LocationID_'+loc in IVs:
			df['LocationID_'+loc] = 1
		lt = d_rf_bact[d_rf_bact.LocationID==loc].LocationTypeID.unique()[0]
		if 'LocationTypeID_'+lt in IVs:
			df['LocationTypeID_'+lt] = 1
		wb = d_rf_bact[d_rf_bact.LocationID==loc].WaterBodyID.unique()[0]
		if 'WaterBodyID_'+wb in IVs:
			df['WaterBodyID_'+wb] = 1
		wt = d_rf_bact[d_rf_bact.LocationID==loc].WaterType.unique()[0]
		if 'WaterType_'+wt in IVs:
			df['WaterType_'+wt] = 1
		## Load hourly flow data
		for col in ['flow_alewife_current', 'flow_alewife_deriv', 'flow_aberjona_current', 'flow_aberjona_deriv']:
			df[col] = df_2016[col].values
		for i in range(len(precip_ts)-1):
			col = 'precip_'+str(precip_ts[i])+'_'+str(precip_ts[i+1])
			df[col] = df_2016[col].values
		## Add this location's data to set
		df['LocationID'] = loc
		df['DateHour'] = df_2016.DateHour.values
		pred_X_loc += [df]

	pred_X = pd.concat(pred_X_loc)
	pred_X.to_csv('test_'+standard+'_2016_prediction_X.csv')
	
	## Model predict values
	model_pred_y = {}
	for model in model_pred_f:
		p_y = model_pred_f[model](pred_X[IVs])
		model_pred_y[model] = pd.DataFrame(data={'DateHour':pred_X['DateHour'], 'exceedence_probability':p_y, 'LocationID':pred_X['LocationID'].values},)
		## Output probability of exceedence
		model_pred_y[model].to_csv('test_'+standard+'_2016_prediction_Y_'+model.replace('\n','_').replace('/','')+'.csv', index=False)
	

	#############################
	## Summary plots
	#############################

	## Predictions
	for model in model_pred_f:
		model_pred_y[model].index = model_pred_y[model].DateHour
		fig, axs = plt.subplots(len(model_pred_y[model].LocationID.unique()), 1, figsize=(4, 13), sharex='all', sharey='all')
		for (loc, group), ax in zip(model_pred_y[model].groupby('LocationID')['exceedence_probability'], axs.flatten()):
			group.plot(kind='line', ax=ax, title=loc, lw=0.5)
		
		plt.tight_layout()
		axs[0].set_ylim(0, 1)
		plt.savefig('test_'+standard+'_2016_prediction_Y_'+model.replace('\n','_').replace('/','')+'.png', 
			  dpi=300, bbox_inches='tight')

	plt.close('all')

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
	plt.gca().set_xlim(50,100)
	plt.xlabel('Accuracy score (%)')
	plt.title('Performance on held out testing set')
	plt.savefig('test_'+standard+'_model_accuracy_comparison.png', bbox_inches='tight')

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
	plt.savefig('test_'+standard+'_model_auc_comparison.png', bbox_inches='tight')


	## ROC curve
	model_scores_pr = {'Baseline\n(Majority rule classifier)':([0,1],[0,1]), 
					'Logistic Regression':auc_calc(clf_lr, xtest=ss_X.transform(X_test))[1:], 
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
	plt.savefig('test_'+standard+'_model_roc_comparison.png', bbox_inches='tight')


	## Feature importance / coefficients
	coefs = {
					'Logistic Regression':pd.Series(clf_lr.coef_[0], name='coefficient', index=IVs), 
					'Logistic Regression\n(w/ interactions)':pd.Series(clf_lri.coef_[0], index=IVs_int, name='coefficient'), 
					'Gradient Boosting': pd.Series(clf_gbm.best_estimator_.feature_importances_, name='feature importance', index=IVs)
					}
	
	for model in coefs:
		ao = np.argsort(coefs[model].values)
		ao = np.array(list(ao[:10]) + list(ao[-10:]))
		
		plt.figure()
		plt.barh(np.arange(len(ao)), coefs[model].values[ao])
		plt.yticks(np.arange(len(ao)), coefs[model].index[ao])
		plt.xlabel(coefs[model].name.capitalize()+' value')
		plt.title(model)
		plt.savefig('test_'+standard+'_'+model.replace(' ','').replace('\n','').replace('/','').replace('(','').replace(')','')+'_coef.png', bbox_inches='tight')




#############################
## Thoughts
#############################

"""
* Should optimize for recall - want to play it safe
"""


#############################
## To-do
#############################

"""

* Use Patsy for custom interacted variable
"""

