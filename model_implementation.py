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

#col_cat = [ 'LocationID', 'WaterType', 'WaterBodyID', 'LocationTypeID', 'Qualifier', ]
col_cat = []

d_h_aberjona = hourly_flow_interp(d_d_aberjona, ['64138_00060_00003'])

flow_dfs = {'h_aberjona':(d_h_aberjona, '64138_00060_00003'), 'h_alewife':(d_h_alewife, '168619_00060')}
col_cat_dummies = prepare_data(
	db_dfs = {'comb':d_combined},
	flow_dfs = flow_dfs,
	l_locationtype = l_locationtype, col_cat = col_cat
	)

df_sim_flow_rainfall, hrs_sim = gen_flow_rainfall('2016-01-01 00:00:00', '2016-12-31 00:00:00', flow_dfs, d_logan['prcp_in'],)

#############################
## Train model
#############################

standard_limits = {
	'Boating':{'ECOLI':1260,'ENT':350},
	'Swimming':{'ECOLI':235,'ENT':104},
	}

model_dic = {}
for model_i in range(len(d_model_list)):
	model_spec = d_model_list.iloc[model_i]
	
	result_db_f, IVs = engineer_features(
		result_db = d_combined, 
		CharacteristicIDs = [model_spec.CharacteristicID],
		Unit = 'MPN/100ml',
		flow_dfs = flow_dfs,
		col_cat_dummies = col_cat_dummies,
		d_logan = d_logan,
		)

	sel = result_db_f.LocationID.isin(model_spec['LocationID(s)'].split( ' + '))
	X_all, y_all, X_train, X_test, y_train, y_test, baseline = gen_train_test(
									result_db_f[sel], 
									{model_spec.CharacteristicID: standard_limits[model_spec.Standard][model_spec.CharacteristicID]}, 
									IVs)

	## Save model
	model_name = '_'.join(model_spec.values)
	model_dic[model_name] = bacteria_model(
		model_spec, 
		X_all, y_all, X_train, X_test, y_train, y_test, 
		IVs, model_spec.Standard, model_spec.CharacteristicID, 'MPN/100ml',
		sensitivity = 0.5, locations = model_spec['LocationID(s)'].split(' + '))
	with open(model_name+'.p', 'w') as f:
		pickle.dump(model_dic[model_name], f)
	



#############################
## Apply model to interpolated rainfall data
#############################

for model_name in model_dic:
	model = model_dic[model_name]
	## Input vector using simulated rainfall and flow data at each location
	pred_X_loc = []
	# Step through locations
	for loc in model.locations:
		df = pd.DataFrame(np.zeros([len(hrs_sim), len(IVs)]), columns = IVs)
		## Load hourly flow data
		for col in df_sim_flow_rainfall:
			df[col] = df_sim_flow_rainfall[col].values
		for i in range(len(precip_ts)-1):
			col = 'precip_'+str(precip_ts[i])+'_'+str(precip_ts[i+1])
			df[col] = df_sim_flow_rainfall[col].values
		## Add this location's data to set
		df['LocationID'] = loc
		df['DateHour'] = df_sim_flow_rainfall.DateHour.values
		pred_X_loc += [df]

	pred_X = pd.concat(pred_X_loc)
	pred_X.to_csv('implemented_'+model_name+'_2016_prediction_X.csv')
	
	## Model predict values
	p_y = model.model.predict_proba(pred_X[IVs])[:, 1]
	model_pred_y = pd.DataFrame(data={'DateHour':pred_X['DateHour'], 'exceedence_probability':p_y, 'LocationID':pred_X['LocationID'].values},)
	## Output probability of exceedence
	model_pred_y.to_csv('test_2016_prediction_Y_'+model_name+'.csv', index=False)

	model_pred_y.index = model_pred_y.DateHour
	fig = plt.figure()
	for (loc, group) in model_pred_y.groupby('LocationID')['exceedence_probability']:
		group.plot(kind='line', ax=plt.gca(), title=model_name, lw=0.5)
	
	plt.tight_layout()
	plt.axhline(model.threshold, label='Threshold', ls='dashed', color='.5')
	plt.legend()
	plt.gca().set_ylim(0, 1)
	plt.savefig('implemented_'+model_name+'_2016_prediction.png', 
			dpi=300, bbox_inches='tight')



#############################
## Diagnostics
#############################

for model_name in model_dic:
	print '\n', model_name
	model_dic[model_name].summarize_performance()


