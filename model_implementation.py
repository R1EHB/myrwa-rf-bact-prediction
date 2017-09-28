from model_implementation_lib import *

#############################
## Load training data
#############################

d_rf, d_mwra_base, d_dcr, d_logan, d_d_aberjona, d_h_alewife, l_locationtype, d_model_list = load_data(
	f_d_rf = 'source_data/rec_flag_2015_16.csv',
	f_d_mwra_base = 'source_data/mwra_base_recflag.csv',
	f_d_dcr = 'source_data/muni_dcr_hist.csv',
	f_d_logan = 'source_data/logan_hourly.csv',
	f_d_d_aberjona = 'source_data/01102500-aberjona-day.txt',
	f_d_h_alewife = 'source_data/01103025-alewife-hour.txt',
	f_l_locationtype = 'source_data/LocationTypeIDs.tsv',
	f_d_model_list = 'RecFlag Model List.xlsx',
	)

d_combined = pd.concat([d_rf, d_mwra_base, d_dcr])

col_cat = [ 'LocationID', 'WaterType', 'WaterBodyID', 'LocationTypeID', 'Qualifier', ] 

d_h_aberjona = hourly_flow_interp(d_d_aberjona, ['64138_00060_00003'])

flow_dfs = {'h_aberjona':(d_h_aberjona, '64138_00060_00003'), 'h_alewife':(d_h_alewife, '168619_00060')}
col_cat_dummies = prepare_data(
	db_dfs = {'comb':d_combined},
	flow_dfs = flow_dfs,
	l_locationtype = l_locationtype, col_cat = col_cat
	)

result_db_f, IVs = engineer_features(
	result_db = d_combined, 
	CharacteristicIDs = ['ECOLI', 'ENT'],
	Unit = 'MPN/100ml',
	flow_dfs = flow_dfs,
	col_cat_dummies = col_cat_dummies,
	d_logan = d_logan,
	)

df_sim_flow_rainfall = gen_flow_rainfall('2016-01-01 00:00:00', '2016-12-31 00:00:00', flow_dfs, d_logan['prcp_in'],)

#############################
## Train model
#############################

standard_limits = {
	'boating':{'ECOLI':1260,'ENT':350},
	'swimming':{'ECOLI':235,'ENT':104},
	}

X_all, y_all, X_train, X_test, y_train, y_test, baseline = gen_train_test(result_db_f, standard_limits['boating'], IVs)

## NOTE: test with class balancing
ss_X, clf_lr = train_model(X_train, y_train)

## Save model
my_model = bacteria_model('test', clf_lr, ss_X, X_all, y_all, X_train, X_test, y_train, y_test, IVs, 'boating', '', 'MPN/100ml')


#############################
## Apply model
#############################

#############################
## Diagnostics
#############################

