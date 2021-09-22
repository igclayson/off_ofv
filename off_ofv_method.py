import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.tools.tools as stools
import math
import itertools
###
'''
Optimal Feature Finder with Optional Functional Variance  (OFF-OFV)
A multi-variable linear regression script that considers all possible linear regressions that can be fitted to a unrestricted N-sized feature and dataset. The regression for each combination of features will result in the fit such that the likelihood that the variance in the dependent data (Y) is described by the regressional fit is maximised, achieved via the Ordinary Least Squares (OLS) method.
Features can optional have their function variants considered as well (specified in the functional_variance_allowed dictionary) such that a set of {f_i} functions of the feature x_j (i.e. f_1(x_j), f_2(x_j), ...) will be also considered.
'''
###


csvs_files={
	"Cu_6MR_PBE_3NN.csv" : "Cu_6MR_PBE_3NN.csv"
#	"toy_model_example.csv" : "toy_model_example.csv"
}
functional_variance_library={ # functions to allow for functional variance. Functions should be linear though else OLS won't work well (non-linear functions require iterative non-exact methods)
	"linear" : lambda x: x,
	"square" : lambda x: x**2,
	"cube" : lambda x: x**3,
	"sqrt" : lambda x: x**(0.5),
	"cbrt" : lambda x: x**(1/3),
	"inverse" : lambda x : 1/(x+1E-6), # note need small constant here to prevent 1/0 evaluations
	"inv sq" : lambda x: 1/(x+1e-6)**2
}
functional_variance_allowed={ # 1 or True will apply functional variance to that feature. "Boolean" features should not have variance applied (i.e. make them 0)
	"volume" : 1,
	"Al_Al_seperation" : 1,
	"average_metal_Al_r" : 1,
	"sigma_metal_Al_r" : 1,
	"minimum_metal_Al_r" : 1,
	"OAl_coord_num" : 0,
	"OSi_coord_num" : 0,
	"OtherO_coord_num" : 0,
	"total_coord_num" : 0,
	"bronsted_num" : 0,
	"square pyramidal" : 0,
	"distorted 6-fold 6 / octahedral" : 0,
	"trigonal bipyrmiada" : 0,
	"square planar" : 0,
	"boat / distorted Td" : 0,
	"trigonal planar" : 0,
	"T-shaped" : 0,
	"mono-atomic species" : 0,
	"atomic unbound" : 0,
	"x": 1,
	"y" : 1,
	"d" : 1,
	"u" : 0
}
print("name&independent_variables&fitted_OLS_coefficients&statistically_significant_variables&standard_deviation_for_x_i&feature_importance_of_x_i&p_value_for_variable_t-tests&R^2_for_model&p_values_for_F_stats&MSE_model")
for csv_file in csvs_files:
	df = pd.read_csv(csvs_files[csv_file], header =0)
	X=df.drop(columns=['step','Relative energy / kJ mol^-1'])
	Y=df.iloc[:, -1] # the target variable must be on the right-most side of the csv
	X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42) # 2/3 train and 1/3 test
	possible_dependent_vars=X_train.columns.values
	possible_dependent_vars_num=len(possible_dependent_vars)
	column_nums=[i for i in range(possible_dependent_vars_num)]
	dependent_variable_combinations=[itertools.combinations(column_nums, i) for i in range(1,possible_dependent_vars_num+1)] # makes an list of iter combination objects (to save space and time) for each possible combination of independent variables
	ranking=[]
	more_likely_fit=1000 # more_likely_fit or more accurate is defined here as the total MSE
	lowest_index=0
	index=0
	p_tests, params, std_array, feature_importance, stat_significant_params =[], [], [], [], [] # need to be here to make sure these arrays are within the scope of the final "best" results printing
	lowest_functional_variance_columns, lowest_function_sequence, lowest_raw_column= [], [], [] # need to be here to make sure these arrays are within the scope of the final "best" results printing
	# Main loop
	# This is currently set to print the parameters and variables everytime a new "best best" fit is found
	for dependent_variable_combination_object in dependent_variable_combinations:
		dependent_variable_combination=list(dependent_variable_combination_object) # note the object is not the actual combinatinon and must be made into a list (list has the column labels)
		for permutation in dependent_variable_combination: # permutation (or combination) is the list of the independent variables we are considering for this run
			column, functional_variance_columns=[], []
			for element in permutation[:]:
				column.append(possible_dependent_vars[element])
				if functional_variance_allowed[possible_dependent_vars[element]]==1: # so if the independent variable has the functional variance flag set to true
					functional_variance_columns.append(possible_dependent_vars[element])
				# functional_variance_columns is the list of features in the current feature combination where we will then apply the functional variance
			feature_number=len(column) # how many features are in this run?
			if len(functional_variance_columns) > 0: # i.e. do we need to bother with functional variance?
				num_of_variable_columns=len(functional_variance_columns) 
				for function_sequence_object in itertools.combinations(functional_variance_library,num_of_variable_columns):
				# the function_sequence object is the list of functions will are applying to this set of functional_variance_columns (e.g. if the function_sequence=['linear','square'] for functional_variance_columns=['x','y'] then we will examine ['linear x','square y'] where this is distinct fit to ['square','linear'] with ['x','y']
					function_sequence=list(function_sequence_object)
					j=0
					X_train_instance=stools.add_constant(X_train[column]) # constants are considered
					column_varied=column[:] # note that column is unchanged until we have considered all functional variances
					while j < num_of_variable_columns: # apply the functional variance the the features where functional variance is allowed
						X_train_instance[functional_variance_columns[j]]=functional_variance_library[function_sequence[j]](X_train_instance[functional_variance_columns[j]])
						column_varied=[w.replace(functional_variance_columns[j],"{} {}".format(function_sequence[j],functional_variance_columns[j])) for w in column_varied]
						X_train_instance.rename({functional_variance_columns[j]:"{} {}".format(function_sequence[j],functional_variance_columns[j])}, axis=1, inplace=True)
						j+=1
					# note this workflow is necessary to prevent comparisons such as (x_1) and (x_1)^2 as these will be correlated dependent variables and invalidate OLS
					model = sm.OLS(Y_train,X_train_instance)
					results = model.fit()
					ranking.append([column_varied,results.params,results.pvalues,results.rsquared,results.f_pvalue,results.mse_resid]) # stores the fit and key information (was necessary for debugging but should be removed)
					if more_likely_fit > results.mse_resid: # the "best best" fit criterion is where the MSE of the residual is reduced
						more_likely_fit=results.mse_resid
						lowest_index=index
						lowest_rank=ranking[lowest_index] # this "lowest rank" is the current "best best" fit and is the only bit of data return from this procedure
						lowest_column=lowest_rank[0]
						lowest_functional_variance_columns=functional_variance_columns
						lowest_function_sequence=function_sequence
						lowest_raw_column=column
						p_tests, params, std_array, feature_importance, stat_significant_params =[lowest_rank[2][0]], [lowest_rank[1][0]], [], [], []
						for i, std in enumerate(X_train_instance[lowest_column].std()):
							 p_tests.append(lowest_rank[2][i])
							 if lowest_rank[2][i] < 0.05:
								 stat_significant_params.append(lowest_column[i])
							 params.append(lowest_rank[1][i])
							 std_array.append(std)
							 feature_importance.append(abs(lowest_rank[1][i]*std))
						print(ranking[lowest_index][1])
					index+=1
			else:
				# if there is no functional variance to consider then a simpler approach can be done
				X_train_instance=stools.add_constant(X_train[column])
				model = sm.OLS(Y_train,X_train_instance)
				results = model.fit()
				ranking.append([column,results.params,results.pvalues,results.rsquared,results.f_pvalue,results.mse_resid])
				if more_likely_fit > results.mse_resid:
					more_likely_fit=results.mse_resid
					lowest_index=index
					lowest_rank=ranking[lowest_index]
					lowest_column=lowest_rank[0]
					p_tests, params, std_array, feature_importance, stat_significant_params =[lowest_rank[2][0]], [lowest_rank[1][0]], [], [], []
					for i, std in enumerate(X_train_instance[lowest_column].std()):
						 p_tests.append(lowest_rank[2][i])
						 if lowest_rank[2][i] < 0.05:
							 stat_significant_params.append(lowest_column[i])
						 params.append(lowest_rank[1][i])
						 std_array.append(std)
						 feature_importance.append(abs(lowest_rank[1][i]*std))
					print(ranking[lowest_index][1])
				index+=1
	lowest_rank=ranking[lowest_index]
	lowest_column=lowest_rank[0]
	# this controls the final data print for export
	print(csv_file,end='&')
	print(lowest_column,end='&')
	print(params, end='&')
	print(stat_significant_params,end='&')
	print(std_array,end='&')
	print(feature_importance,end='&')
	print(p_tests,end='&')
	print(lowest_rank[3], end='&')
	print(lowest_rank[4],end='&')
	print(lowest_rank[5], end='&')
	print("")
	print("=======")
	print("Lowest MSE parameters and cofficients:\n")
	print(lowest_rank[1])
	# currently haven't coded testing of the test data and respective analsysis
	print("=======")
	print("Testing data:\n")
	print(X_test)
	print(Y_test)
