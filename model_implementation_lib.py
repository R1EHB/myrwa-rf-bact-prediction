import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc

plt.ion()

#############################
## Load training data
#############################

def load_data(f_d_rf, f_d_mwra_base, f_d_dcr, f_d_logan, f_d_d_aberjona, f_d_h_alewife, f_l_locationtype, f_d_model_list):
	"""
	Load relevant datasets given filenames specified as input
	"""
	print "Loading data"
	
	## Load main dataset
	d_rf = pd.read_csv(f_d_rf, parse_dates=['DateHour'])

	## Load MWRA historical data
	d_mwra_base = pd.read_csv(f_d_mwra_base, parse_dates=['DateHour'])

	## Load DCR beach historical data
	d_dcr = pd.read_csv(f_d_dcr, parse_dates=['DateHour'])

	## Load rainfall data
	d_logan = pd.read_csv(f_d_logan, index_col='datetime', parse_dates=True)

	## Load streamflow data
	d_d_aberjona = pd.read_table(f_d_d_aberjona, comment='#', index_col='datetime').iloc[1:]
	d_h_alewife = pd.read_table(f_d_h_alewife, comment='#', index_col='datetime').iloc[1:]
	d_d_aberjona.index = pd.to_datetime(d_d_aberjona.index) # parse_dates does not work for this date format
	d_h_alewife.index = pd.to_datetime(d_h_alewife.index)

	## Lookup tables
	l_locationtype = pd.read_table(f_l_locationtype, sep='\t', index_col = 'ID')
	
	## Model 
	d_model_list = pd.read_excel(f_d_model_list)
	
	return d_rf, d_mwra_base, d_dcr, d_logan, d_d_aberjona, d_h_alewife, l_locationtype, d_model_list

#############################
## Transform data
#############################

def hourly_flow_interp(d_d, cols):
	"""
	Generate interpolated hourly streamflow from daily data
	
	Takes dataframe of streamflow data indexed on datetime as input and a list of columns (site names) to generate flow interpolations for.
	
	Outputs equivalent dataframe interpolated to hourly resolution
	"""
	print "Interpolating streamflow data"
	
	hour_series = pd.date_range(np.min(d_d.index.values), np.max(d_d.index.values), freq=pd.Timedelta(1, 'h'))
	d_h = d_d.loc[hour_series]
	for col in ['agency_cd','site_no']:
		d_h[col] = d_h[col].fillna(method='ffill')

	d_h['datetime'] = d_h.index
	
	for col in cols:
		d_h[col] = d_h[col].astype(float).interpolate()
	
	return d_h



def prepare_data(db_dfs, flow_dfs, l_locationtype, col_cat):
	"""
	Prepare data for feature engineering
	
	Expects two dictionaries as input -
	* db_dfs: Dictionary containing all result databases as dataframes.  Keys can be anything; values are dataframes.
	* flow_dfs: Dictionary containing all HOURLY flow databases as dataframes.  Keys should be the waterbody name; values should be (dataframe, column) pairs, where the column is the one containing relevant streamflow data.
	
	The dataframes in both these dictionaries will be transformed in place (the input dictionary will be edited directly)
	
	col_cat is a listing of categorical variable columns that will be transformed into dummy columns in the dataframes.
	
	Returns col_cat_dummies, list of dummy categorical variables to pass to engineer_features
	"""
	print "Preparing data"

	## Times
	for name in db_dfs:
		db_dfs[name].DateHour = pd.to_datetime(db_dfs[name].DateHour)

	## Not needed - using parse_dates in read_table
	#for name in flow_dfs:
		#flow_dfs[name].datetime = pd.to_datetime(flow_dfs[name].datetime)

	#d_logan.datetime = pd.to_datetime(d_logan.datetime)
	#d_logan.index = d_logan.datetime

	## LocationTypeID names
	for name in db_dfs:
		db_dfs[name]['LocationTypeID'] = db_dfs[name]['LocationTypeID'].apply(lambda x: l_locationtype['LocationTypeName'].loc[x]).values

	col_cat_dummies = []
	for name in db_dfs:
		for col in col_cat:
			df_dum = pd.get_dummies(db_dfs[name][col], prefix=col, drop_first=1, dummy_na=1)
			for new_col in df_dum.columns:
				db_dfs[name][new_col] = df_dum[new_col]
			col_cat_dummies += df_dum.columns.tolist()

	col_cat_dummies = np.unique(col_cat_dummies)

	## Calculate streamflow derivative
	for key in flow_dfs:
		df, col = flow_dfs[key]
		df['flow_deriv'] = np.nan
		df['flow_deriv'].loc[1:] = df[col].astype(float).values[1:] - df[col].astype(float).values[:-1]
	
	return col_cat_dummies

def gen_flow_rainfall(start_time, end_time, flow_dfs, rain_series, freq = (4, 'h'), precip_ts = [0,12,24,48,72,96,120]):
	"""
	Generate flow and rainfall data between some start_time and end_time for simulation purposes
	
	Parameters:
	* start_time: Start point for simulated range
	* end_time: End point for simulated range
	* flow_dfs: dictionary in the same format as that passed to prepare_data
	* rain_series is rainfall data to be passed to sum_rain
	* freq: (numeric, unit) tuple specifying frequency interval for simulated data range, to be passed to pd.Timedelta
	* precip_ts: Defaults to same list as engineer_features - needs to match
	"""
	print "Generating flow and rainfall data for simulation"
	
	hrs_sim = pd.date_range(pd.Timestamp(start_time), pd.Timestamp(end_time), freq=pd.Timedelta(*freq))
	df_sim = pd.DataFrame(data = {'DateHour':hrs_sim.values})
	
	for key in flow_dfs:
		df, col = flow_dfs[key]
		## Load hourly flow data
		df_sim['flow_'+key+'_current'] = df[col].loc[hrs_sim].values
		df_sim['flow_'+key+'_deriv'] = df['flow_deriv'].loc[hrs_sim].values
		## Missing some values, so interpolate
		df_sim['flow_'+key+'_current'].interpolate('linear', inplace=True)
		df_sim['flow_'+key+'_current'].fillna(0, inplace=True)
		df_sim['flow_'+key+'_deriv'].interpolate('linear', inplace=True)
		df_sim['flow_'+key+'_deriv'].fillna(0, inplace=True)
	
	## Calculate hourly precipitation
	for i in range(len(precip_ts)-1):
		df_sim['precip_'+str(precip_ts[i])+'_'+str(precip_ts[i+1])] = \
			df_sim['DateHour'].apply(lambda x: sum_rain(x, [precip_ts[i], precip_ts[i+1]], rain_series))
	return df_sim


## Join rainfall data on different timescales
def sum_rain(timestamp, hours, rain_series):
	sel = (rain_series.index >= timestamp - pd.Timedelta(hours[1], 'h')) & (rain_series.index < timestamp - pd.Timedelta(hours[0], 'h'))
	return rain_series[sel].sum()


def engineer_features(result_db, CharacteristicIDs, Unit, flow_dfs, col_cat_dummies, d_logan, 
					  precip_ts = [0,12,24,48,72,96,120], DV = 'ResultValue'):
	"""
	Transform prepared data to engineer features for modeling
	
	Parameters:
	* result_db is the results database to operate on 
	* CharacteristicID is a list of measurement types to filter on units (e.g. ['ECOLI']) , and...
	* Unit is the unit to select.
	* flow_dfs is a dictionary in the same format as that passed to prepare_data
	* col_cat_dummies is the list of categorical dummy variables output by prepare_data
	* d_logan: Rainfall dataframe
	
	Non-required parameters are,
	* precip_ts is the series of rainfall timescales to calculate using sum_rain
	* DV is the dependent variable name in result_db
	
	This function also filters out rows with non_null 'FlagID' fields in the 
	"""
	print "Engineering features"
	
	## Filter to CharactericID=='ECOLI' and Units=='MPN/100ml'
	sel = np.any(
		[(result_db["CharacteristicID"]==char) & (result_db['Units']==Unit) for char in CharacteristicIDs],
			axis = 0)
	result_db_f = result_db[sel]

	## Filter out flagged rows
	result_db_f = result_db_f[result_db_f['FlagID'].isnull()]

	## Join streamflow data and derivatives
	flow_vars = []
	for key in flow_dfs:
			df, col = flow_dfs[key]
			result_db_f['flow_'+key+'_current'] = df[col].loc[result_db_f.DateHour].values
			result_db_f['flow_'+key+'_deriv'] = df['flow_deriv'].loc[result_db_f.DateHour].values
			flow_vars += ['flow_'+key+'_current', 'flow_'+key+'_deriv']

	for i in range(len(precip_ts)-1):
		result_db_f['precip_'+str(precip_ts[i])+'_'+str(precip_ts[i+1])] = result_db_f.DateHour.apply(
			lambda x: sum_rain(x, [precip_ts[i], precip_ts[i+1]], d_logan['prcp_in'])
			)

	## Gather variables
	IVs = flow_vars + list(col_cat_dummies) + ['precip_'+str(precip_ts[i])+'_'+str(precip_ts[i+1]) for i in range(len(precip_ts)-1)]
	#IVs = [iv for iv in IVs if iv in result_db_f] # some columns came from other dataframes - eliminate them
	IVs = [iv for iv in IVs if iv.startswith('Qualifier')==0] # does not make causal sense to include the bacterial results qualifier
	
	return result_db_f, IVs


def gen_train_test(result_db_f, standards_dic, IVs, DV = 'ResultValue', test_size=0.2, random_state=101):
	"""
	Generate training and testing data matrices
	
	Parameters:
	* result_db_f: Output from engineer_features
	* standards_dic is a dictionary with CharacteristicIDs as keys and the relevant swimming or boating standard level as values (must match Unit passed to engineer_features)
	* IVs is the independent variable list from engineer_features
	* DV is the dependent variable name in result_db and should match that passed to engineer_features
	* test_size: Fraction specifying size of test set
	* random_state: Set this to fix the sate of the random number generator
	"""
	print "Splitting train and test data"
	
	# Instantiate matrices
	X = result_db_f.loc[:,IVs].astype(float).values
	Y = np.any(
		[((result_db_f[DV].values > standards_dic[key]) & (result_db_f['CharacteristicID'] == key)) for key in standards_dic]
		, axis =0).astype(float)
	
	## Turns out some flow values are null.  Fill with mean
	imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
	imp.fit(X)
	X_imp = imp.transform(X)

	X_train, X_test, y_train, y_test = \
		train_test_split(X_imp, Y, test_size=test_size, random_state=random_state, stratify=Y)

	baseline = 1-y_test.mean()
	
	return X_imp, Y, X_train, X_test, y_train, y_test, baseline



#############################
## Train model
#############################

def train_model(X_train, y_train, Cs = 10, n_jobs = 4, cv = 7, class_weight=None):
	"""
	Train the logistic regression model, using sklearn's LogisticRegressionCV
	
	Parameters:
	* Cs: Number of regularization parameters to test in grid search
	* n_jobs: Number of CPU cores to parallelize over
	* cv: Number of cross validation folds to use
	* class_weight: Class weighting value to pass (None by default; may also want to use 'balanced')
	"""
	print "Fitting model"

	ss_X = StandardScaler().fit(X_train)

	clf_lr = LogisticRegressionCV(Cs=10, n_jobs=4, cv=7, class_weight=class_weight).fit(ss_X.transform(X_train), y_train)

	### With interactions
	#pf_int = PolynomialFeatures(degree=2)
	#pf_int.fit(ss_X.transform(X_train))
	
	#IVs_int = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(IVs,p) for p in pf_int.powers_]]

	#clf_lri = LogisticRegressionCV(Cs=10, n_jobs=4, cv=7).fit(pf_int.transform(ss_X.transform(X_train)), y_train)

	return ss_X, clf_lr



#############################
## Bacteria prediction model class
#############################

def auc_calc(model, ytest, xtest):
		fpr, tpr, thresholds = roc_curve(ytest, model.predict_proba(xtest)[:, 1])
		return auc(fpr, tpr), fpr, tpr

class bacteria_model():
	"""
	"""
	
	def __init__(self, name, X_all, y_all, X_train, X_test, y_train, y_test, IVs, 
			  standard, CharacteristicID, Unit,
			  DV = 'ResultValue', precip_ts = [0,12,24,48,72,96,120], 
			  model_Cs = 10, model_n_jobs = 4, model_cv = 7, model_class_weight=None):
		self.name = name
		self.model = None
		self.standardizer = None
		self.X_all = X_all
		self.y_all = y_all
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
		self.IVs = IVs
		self.standard = standard
		self.CharacteristicID = CharacteristicID
		self.Unit = Unit
		self.DV = DV
		self.precip_ts = precip_ts
		self.Cs = model_Cs
		self.n_jobs = model_n_jobs
		self.cv = model_cv
		self.class_weight = model_class_weight
		
		## Train model with all data by default
		self.train_model(X_all, y_all, 
				   Cs = self.Cs, n_jobs = self.n_jobs, cv = self.cv, class_weight=self.class_weight)
	
	def train_model(self, X, Y, **kwargs):
		self.standardizer, self.model = train_model(X, Y, **kwargs)
	
	def retrain_model(self, X, Y):
		return self.model.fit(X, Y)
	
	def predict(self, X):
		return self.model.predict(self.standardizer.transform(X))
	
	def get_auc(self):
		"""
		Get AUC on held out testing data
		"""
		if 'auc' not in self.__dict__:
			## Retrain on testing data if not already done
			if 'model_train_test' not in self.__dict__:
				self.model_train_test = self.retrain_model(self.X_train, self.y_train)
			
			self.auc = auc_calc(self.model_train_test, self.y_test, self.X_test)
		
		return self.auc

def load_model(f):
	return pickle.load(f)


"""
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
for loc in result_db_f.LocationID.unique():
	df = pd.DataFrame(np.zeros([len(hrs_2016), len(IVs)]), columns = IVs)
	if 'LocationID_'+loc in IVs:
		df['LocationID_'+loc] = 1
	lt = result_db_f[result_db_f.LocationID==loc].LocationTypeID.unique()[0]
	if 'LocationTypeID_'+lt in IVs:
		df['LocationTypeID_'+lt] = 1
	wb = result_db_f[result_db_f.LocationID==loc].WaterBodyID.unique()[0]
	if 'WaterBodyID_'+wb in IVs:
		df['WaterBodyID_'+wb] = 1
	wt = result_db_f[result_db_f.LocationID==loc].WaterType.unique()[0]
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


"""
