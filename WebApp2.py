import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#######################################################################################################

import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from missingpy import MissForest
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from feature_engine.encoding import OneHotEncoder
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures, DropDuplicateFeatures
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

#########################################################################################################

st.title('Machine Learning Feature Engineering Web Application')
st.write('''This web app is used for preprocessing a dataset, feature engineering, dimensionality reduction, hyperparameter tuning, and modeling.
	There are many opions that could have been implemented in terms of algorithms, hyperparameters, etc.  However, this is a demonstration 
	rather than an all inclusive model solution.  This web app shows the ability to input data, process data, select features, reduce high dimensionality,
	  cross validate, and test the final model.''')
st.write('Please visit my LinkedIn for more information about myself: www.linkedin.com/in/brandon-johnson-09645ba9')

#########################################################################################################

def main():
	
###################################################################### UPLOADING DATA ####################################################################

	st.subheader('Uploading Data')

	data = st.file_uploader('Please upload a dataset in csv format', type=['csv'])

	if st.checkbox('No data? Click here and choose from the datasets available'):
		dataset_list = ['Iris Classification', 'Car Regression']
		selected_data = st.selectbox('Choose which dataset to use:', dataset_list)

		@st.cache(suppress_st_warning=True, allow_output_mutation=True)
		def get_data(selected_data1):
			if selected_data1 == 'Iris Classification':
				st.info('This dataset has had values removed from the feature column for the purpose of imputing.')
				data = 'https://raw.githubusercontent.com/SPDA36/WebApp_Missing_Data/main/iris_missing_values.csv'
			if selected_data1 == 'Car Regression':
				st.info('This dataset is complete and contains car features and car prices but will need feature elemination due to high cardinality')
				data = 'https://raw.githubusercontent.com/SPDA36/WebApp_Feature_Engineering/main/car%20data%20all%20together.csv'
			return data
		data = get_data(selected_data)


################################################################################ CONVERTING DATA TO DATAFRAME ###########################

	if data is not None:
		st.success('Data Successfully Loaded')
		st.warning('''WARNING: If you select a checkbox and then unselect a checkbox,
			  the unselection of the checkbox will undo all changes made under said checkbox.  Also, if you need to make any changes that requires you
				to work backwards, please deselect checkboxes as you move backwards.  This will prevent processes from conflicting with each other''')

		@st.cache(suppress_st_warning=True, allow_output_mutation=True)
		def get_df1(data1):
			df = pd.read_csv(data1)
			return df
		df1 = get_df1(data)
		st.write('Uploaded dataset (<NA> represents missing values):')
		st.dataframe(df1)

		st.write('_________________________________________________________________________________________________________________')
			  ##################### PROBLEM TYPE ############################

		st.subheader('Select Problem Type')
		type_problem_list = ['Classification', 'Regression']
		type_problem = st.selectbox('Is this a classification or regression problem?  If using built in datasets, then the problem type is listed in their name', type_problem_list)
		
		st.write('_________________________________________________________________________________________________________________')

		st.subheader('Feature Data Types')
		if st.checkbox('Click here to show data types'):
			st.dataframe(df1.dtypes.astype(str))

			st.info('''NOTE: Notice the data types above.  If a features is not the correct data type, then that will need to be changed.
			Missing values default to np.nan values.  You will have options to drop or impute missing values values.  Data types that are incorrect will 
			need to be replaced with correct data types.  For instance, if you know a feature is supposed to be an float type but it is 
			showing that it is an O type (object), that means there are string characters in the feature.  Maybe there is a dollar 
			sign that needs to be removed.  This web app assumes that data types are corrected in the data querying process prior to using this app.  However, the web app allows for missing data to be imputed or dropped.''')
		st.write('_________________________________________________________________________________________________________________')

##################################################################### TARGET VARIABLE ######################################

		st.subheader('y Target Variable Selection and X Feature Selection')
		st.warning('WARNING: DO NOT SKIP THIS SECTION')
		selected_y = st.selectbox('Please select the traget variable:', df1.columns)
		selected_X = st.multiselect('Please select all features to include in the X set for feature engineering:', df1.drop(selected_y, axis=1).columns)
		y = df1[selected_y]
		X = df1[selected_X]

		if type_problem == 'Classification':
			if st.checkbox('Show target class balance?'):
				y_balance = pd.DataFrame(y.value_counts())
				y_balance['proportions'] = round(y_balance.div(y_balance.count()),3)
				y_balance.reset_index(inplace=True)
				y_balance.columns = ['target', 'count', 'proportions']
				st.write(y_balance)
		st.write('_________________________________________________________________________________________________________________')


################################################################# TRAIN TEST SPLIT #########################################

		st.subheader('Train Test Split')
		st.write('Please select a training and testing split proportion and the random state:')
		
		st.write('X shape: ', X.shape)
		st.write('y shape: ', y.shape)

		test_size = st.slider('Select the testing proportion. Default is 0.20', min_value=0.1, max_value=0.5, step=0.05, value=0.2)
		rand_state = st.slider('Select random state. Any value is accpetable', min_value=0, max_value=1000, step=1, value=10)

		@st.cache(suppress_st_warning=True, allow_output_mutation=True)
		def get_tts(X1,y1,test_size1,rand_state1):
			X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=test_size1, random_state=rand_state1)
			return X_train1, X_test1, y_train1, y_test1
		X_train, X_test, y_train, y_test = get_tts(X,y,test_size,rand_state)

		st.write('X_train shape: ',X_train.shape)
		st.write('y_train shape: ', y_train.shape) 
		st.write('X_test shape: ', X_test.shape)
		st.write('y_test shape: ', y_test.shape)

		st.write('_________________________________________________________________________________________________________________')



########################################################################## ENCODING CATEGORICAL FEATURES ###########################################

		st.subheader('Encoding Categorical Features')

		if st.checkbox('Click here if you have categorical features that need encoding.  If you change your mind, unclick this checkbox'):
			st.dataframe(X_train)
			############ NOMINAL ##############

			st.write('')
			st.write('')
			st.info('Only select Nominal features here:')
			if st.checkbox('Click here if you have a nominal feature to encode'):
				selected_encode = st.multiselect('Please select all of the nominal features needing to be encoded into their own feature', X_train.columns)
				if st.checkbox('Please click checkbox after selecting nominal features'):
					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def get_one_hot(X_train1,X_test1):
						one_hot = OneHotEncoder(variables = selected_encode)
						X_train2 = one_hot.fit_transform(X_train1)
						X_test2 = one_hot.transform(X_test1)
						return X_train2, X_test2
					X_train,X_test = get_one_hot(X_train,X_test)
				
					st.write('X_train Encoded Set:')
					st.write('X_train shape', X_train.shape)
					st.dataframe(X_train)
					st.write('X_test Encoded Set:')
					st.write('X_test shape: ', X_test.shape)
					st.dataframe(X_test)
			
			
			########### ORDINAL ###############

			st.write('')
			st.write('')
			st.info('Only select ordianl features here:')
			if st.checkbox('Click here if you have an ordinal feature to encode'):
				selected_ordinal = st.selectbox('Please select the ordinal features to encode', X.columns)
				ordinal_order = st.multiselect('Please select all of the variables in ranking order.  Starting with the lowest rank in the order', X[selected_ordinal].unique())

				if st.checkbox('Please click checkbox after selecting the ranking of the ordinal feature'):

					ord_encoder = OrdinalEncoder(categories=[ordinal_order])
					ord_encoded_X_train = ord_encoder.fit_transform(X_train[[selected_ordinal]])
					ord_encoded_X_train = pd.DataFrame(ord_encoded_X_train, columns=[selected_ordinal+'_ord'], index=X_train.index)
					X_train = pd.concat([X_train.drop(selected_ordinal, axis=1), ord_encoded_X_train], axis=1)

					st.write('X_test Ordinal Encoded:')
					st.dataframe(X_train)

					ord_encoded_X_test = ord_encoder.transform(X_test[[selected_ordinal]])
					ord_encoded_X_test = pd.DataFrame(ord_encoded_X_test, columns=[selected_ordinal+'_ord'], index=X_test.index)
					X_test = pd.concat([X_test.drop(selected_ordinal, axis=1), ord_encoded_X_test], axis=1)

					st.write('X_test Ordinal Encoded:')
					st.dataframe(X_test)

				if st.checkbox('Click here if you have another ordinal feature to encode'):
					selected_ordinal1 = st.selectbox('Please select the ordinal features to encode', X_train.columns, key=1) # HAVE TO USE key= TO GIVE THE CHECKBOXES AND MULTISELECT BOXES THEIR OWN UNIQUE ID
					ordinal_order1 = st.multiselect('Please select all of the variables in ranking order.  Starting with the lowest rank in the order', X_train[selected_ordinal1].unique(), key=2)
					
					if st.checkbox('Please click checkbox after selecting the ranking of the ordinal feature', key=3):
						ord_encoder1 = OrdinalEncoder(categories=[ordinal_order1])
						ord_encoded_X_train1 = ord_encoder1.fit_transform(X_train[[selected_ordinal1]])
						ord_encoded_X_train1 = pd.DataFrame(ord_encoded_X_train1, columns=[selected_ordinal1+'_ord'], index=X_train.index)
						X_train = pd.concat([X_train.drop(selected_ordinal1, axis=1), ord_encoded_X_train1], axis=1)

						st.write('X_test Ordinal Encoded:')
						st.dataframe(X_train)

						ord_encoded_X_test1 = ord_encoder1.transform(X_test[[selected_ordinal1]])
						ord_encoded_X_test1 = pd.DataFrame(ord_encoded_X_test1, columns=[selected_ordinal1+'_ord'], index=X_test.index)
						X_test = pd.concat([X_test.drop(selected_ordinal1, axis=1), ord_encoded_X_test1], axis=1)

						st.write('X_test Ordinal Encoded:')
						st.dataframe(X_test)

		st.write('_________________________________________________________________________________________________________________')


##################################################################### NULL VALUES ##################################################

		st.subheader('Null/Missing Values')
		if st.checkbox('Show null/missing values? Click the checkbox'):
			st.write('X Training Set:')
			X_train_null = X_train.isnull().sum()
			X_train_null = pd.DataFrame(X_train_null, columns=['Number of null values per column'])
			st.dataframe(X_train_null)

			st.write('y Training Set:')
			y_train_null = y_train.isnull().sum()
			y_train_null = pd.DataFrame([y_train_null], columns=['Number of null values'])
			st.dataframe(y_train_null)

			st.write('X Testing Set:')
			X_test_null = X_test.isnull().sum()
			X_test_null = pd.DataFrame(X_test_null, columns=['Number of null values per column'])
			st.dataframe(X_test_null)

			st.write('y Testing Set:')
			y_test_null = y_test.isnull().sum()
			y_test_null = pd.DataFrame([y_test_null], columns=['Number of null values'])
			st.dataframe(y_test_null)

		st.write('_________________________________________________________________________________________________________________')

######################################################################### IMPUTER SELECTION ##############################################


		st.subheader('Imputing Missing Continious Values')
		if st.checkbox('Need to impute missing values?  Click the checkbox'):
			selected_col_impute = st.multiselect('Select all numerical features that need to be imputed',X.columns,default=None)

			if st.checkbox('Click here to impute missing values after selecting the features above'):

				imputer_options = ['Missing Forest Imputer', 'KNN Imputer']
				selected_imputer =  st.selectbox('Please select the desired imputer from the list', imputer_options)

				if st.checkbox('Click here after selecting the imputer from the list above'):

		########################### MISSING FOREST IMPUTER #################################


					if selected_imputer == 'Missing Forest Imputer':
						imputer = MissForest(criterion='squared_error', max_features=1.0)

						@st.cache(suppress_st_warning=True, allow_output_mutation=True)
						def get_mf_impute_train(X_train1,selected_col_impute1, imputer1):
							imputed_X_train = imputer1.fit_transform(X_train1[selected_col_impute1])
							imputed_X_train = pd.DataFrame(imputed_X_train, columns=selected_col_impute1, index=X_train1.index)
							X_train2 = pd.concat([imputed_X_train, X_train1.drop(selected_col_impute1, axis=1)], axis=1)
							return X_train2, imputed_X_train, imputer
						X_train, imputed_X_train, imputer = get_mf_impute_train(X_train, selected_col_impute, imputer)

						st.write('Imputed X_train Dataset:')
						st.write('Imputed X_train shape: ', X_train.shape)
						st.dataframe(X_train)

						@st.cache(suppress_st_warning=True, allow_output_mutation=True)
						def get_mf_impute_test(X_test1, selected_col_impute1, imputer1):
							imputed_X_test = imputer1.transform(X_test1[selected_col_impute1])
							imputed_X_test = pd.DataFrame(imputed_X_test, columns=selected_col_impute1, index=X_test1.index)
							X_test2 = pd.concat([imputed_X_test, X_test1.drop(selected_col_impute1, axis=1)], axis=1)
							return X_test2
						X_test = get_mf_impute_test(X_test, selected_col_impute, imputer)

						st.write('Imputed X_test Dataset:')
						st.write('Imputed X_test shape: ',X_test.shape)
						st.dataframe(X_test)

		############################# KNN IMPUTER ############################################

					if selected_imputer == 'KNN Imputer':

						k = st.slider('Select the k-value by moving the slider', min_value=1, max_value=100, value=15, step=1)
						st.info('General guidance for selecting a k-value is based on the square root of the number of observations and then divide by 2. This is not always perfect which is a disadvantage of using KNN for imputing.  Knowing which k-value to use can be tricky.')
						st.write('Suggested k-value: ', round(np.sqrt(X_train.shape[0])/2))
						imputer = KNNImputer(n_neighbors= k)

						@st.cache(suppress_st_warning=True, allow_output_mutation=True)
						def get_knn_impute_train(X_train1, imputer1, selected_col_impute1):
							imputed_X_train = imputer1.fit_transform(X_train1[selected_col_impute1])
							imputed_X_train = pd.DataFrame(imputed_X_train, columns=selected_col_impute1, index=X_train1.index)
							X_train2 = pd.concat([imputed_X_train, X_train1.drop(selected_col_impute1, axis=1)], axis=1)
							return X_train2, imputed_X_train, imputer
						X_train, imputed_X_train, imputer = get_knn_impute_train(X_train, imputer, selected_col_impute)

						st.write('Imputed X_train Dataset:')
						st.write('Imputed X_train shape: ',X_train.shape)
						st.dataframe(X_train)

						@st.cache(suppress_st_warning=True, allow_output_mutation=True)
						def get_knn_impute_test(X_test1, selected_col_impute1):
							imputed_X_test = imputer.transform(X_test1[selected_col_impute1])
							imputed_X_test = pd.DataFrame(imputed_X_test, columns=selected_col_impute1, index=X_test1.index)
							X_test2 = pd.concat([imputed_X_test, X_test1.drop(selected_col_impute1, axis=1)], axis=1)
							return X_test2
						X_test = get_knn_impute_test(X_test, selected_col_impute)

						st.write('Imputed X_test Dataset:')
						st.write('Imputed X_test shape: ',X_test.shape)
						st.dataframe(X_test)

######################################################### ISOLATION FOREST ############################################################################

					st.subheader('Outlier/Anomaly Detection Analysis')

					ift = IsolationForest(n_estimators= 100, max_samples='auto', contamination=0.1, max_features=1.0)

					detecting_X_train = imputed_X_train.copy(deep=True)

					ift.fit(imputed_X_train)

					detecting_X_train['anomaly'] = ift.predict(imputed_X_train)

					detecting_X_train['score'] = ift.decision_function(imputed_X_train)

					st.write('anomaly = -1')
					st.dataframe(detecting_X_train['anomaly'].value_counts())

					outliers = detecting_X_train[detecting_X_train['anomaly'] == -1]
					st.write('Anomaly by record in the X_train set.  Ordered by highest anomaly to lowest anomaly')
					st.dataframe(outliers.sort_values(by='score'))

					st.info('It is not always necessary to remove these outliers/anomolies, but knowing that they could impact the model is important')
		st.write('_________________________________________________________________________________________________________________')

##################################################### DROPPING MISSING VALUES ##########################################################

		st.subheader('Dropping Missing Values')
		if st.checkbox('Please check this box if you have missing values that need removed.  Consider imputing the values first'):
			
			if st.checkbox('Drop all missing values?'):
				st.write('This option will drop all records with missing values')
				
				Xy_train = pd.concat([X_train,y_train], axis=1)
				Xy_train.dropna(axis=0, inplace=True)
				X_train = Xy_train.drop(selected_y, axis=1)
				y_train = Xy_train[selected_y]
				st.write('Updated X_train set:')
				st.write('X_train shape: ', X_train.shape)
				st.dataframe(X_train)
				st.write('Updated y_train set:')
				st.write('X_train shape: ', y_train.shape)
				st.dataframe(y_train)

				Xy_test = pd.concat([X_test,y_test], axis=1)
				Xy_test.dropna(axis=0, inplace=True)
				X_test = Xy_test.drop(selected_y, axis=1)
				y_test = Xy_test[selected_y]
				st.write('Updated X_test set:')
				st.write('X_test shape: ', X_test.shape)
				st.dataframe(X_test)
				st.write('Updated y_test set:')
				st.write('X_test shape: ', y_test.shape)
				st.dataframe(y_test)
		st.write('_________________________________________________________________________________________________________________')



################################################################# REDUNDANT CORRELATED DUPLICATED #########################################


		st.subheader('Redundant, Duplicated, Correlated Data Removal')
		if st.checkbox('Please check this box if you have redundant, correlated, or duplicated data'):
			st.write('')
			st.write('')

				#################################### REDUNDANT REMOVAL ###############################

			if st.checkbox('Check this box for redundant data removel'):
				st.subheader('Redundancy Analysis')
				drop_redundant = DropConstantFeatures()
				X_train = drop_redundant.fit_transform(X_train)
				X_test = drop_redundant.transform(X_test)

				st.write('Updated X_train set')
				st.write('X_train shape: ', X_train.shape)
				st.dataframe(X_train)

				st.write('Updated X_test set')
				st.write('X_test shape', X_test.shape)
				st.dataframe(X_test)


					################################# DUPLICATE REMOVAL ##########################################

			if st.checkbox('Check this box for duplicate data removal'):
				st.subheader('Duplicate Analysis')
				drop_duplicates = DropDuplicateFeatures()
				X_train = drop_duplicates.fit_transform(X_train)
				X_test = drop_duplicates.transform(X_test)

				st.write('Updated X_train set')
				st.write('X_train shape: ', X_train.shape)
				st.dataframe(X_train)

				st.write('Updated X_test set')
				st.write('X_test shape', X_test.shape)
				st.dataframe(X_test)


					################## CORRELATED REMOVAL ###########################

			if st.checkbox('Check this box for correlated feature analysis and data removal'):
				st.subheader('Correlation Analysis')
				st.info('NOTE: Simply removing features because they have a high pairwise correlation coefficient does not tell the full story.  This over-simplification can cause your dataset to lose valuable information.  Consider regulariaztion if using a linear model or choose a machine learning algorithm that is robust to high correlated features.')
				if st.checkbox('Show pair plot.  WARNING: If there are many features, it could exceed the limitation of the web application.  Do not use when high cardinality exist'):
					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def get_pairplot(X_train1,y_train1):
						Xy_train = pd.concat([X_train1,y_train1], axis=1)
						fig = sns.pairplot(Xy_train)
						return fig
					fig = get_pairplot(X_train,y_train)
					st.pyplot(fig) 

				corr_method_list = ['pearson', 'spearman']
				corr_method = st.selectbox('Please select the method of correlation detection', corr_method_list)
				st.info('Note: Pearson assumes a linear relationship.  Spearman does not assume linear relationship')
				corr_threshold = st.slider('''Please select the correlation statistic threshold.  This threshold will
					 determine which features to drop if their correlation coefficient exceeds the threshold''',min_value=0.5, max_value=1.0, value=1.0, step=0.02)

				if st.checkbox('Show correlation heatmap? NOTE: If there are many features, this might not display correctly.  Use the resize options'):
					height = st.selectbox('Resize height', range(6,32,2))
					width = st.selectbox('Resize width', range(6,32,2))
					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def get_heatmap(X_train1,y_train1,corr_method1):
						fig2 = plt.figure(figsize=(width,height))
						Xy_train = pd.concat([X_train1,y_train1], axis=1)
						sns.heatmap(Xy_train.corr(method=corr_method1), annot=True, cmap='GnBu')
						return fig2
					fig2 = get_heatmap(X_train,y_train,corr_method)
					st.pyplot(fig2)

				if st.checkbox('Show correlation matrix?'):
					Xy_train = pd.concat([X_train,y_train], axis=1)
					st.dataframe(Xy_train.corr(method=corr_method))

				if st.checkbox('Check this box to drop correleated features based on the threshold seleted above?'):
					drop_corr = DropCorrelatedFeatures(method=corr_method, threshold=corr_threshold)
					X_train = drop_corr.fit_transform(X_train)
					X_test = drop_corr.transform(X_test)

					st.write('Updated X_train set:')
					st.write('X_train shape: ', X_train.shape)
					st.dataframe(X_train)

					st.write('Updated X_test set:')
					st.write('X_test shape', X_test.shape)
					st.dataframe(X_test)

		st.write('_________________________________________________________________________________________________________________')


######################################################### SCALING DATA ############################################################################

		st.subheader('Scale Data')

		if st.checkbox('Need to scale data?  Click the checkbox'):
			scaler_list = ['Standard Scaler', 'Min-Max Scaler', 'Robust Scaler']
			selected_scaler = st.selectbox('Please select a scaler from the list', scaler_list)
			st.dataframe(X_train)

					##################################### STANDARD SCALER ##############################

			if selected_scaler == 'Standard Scaler':
				scaler = StandardScaler()

				selected_columns_to_scale = st.multiselect('Please select continuous variable features to scale', X_train.columns)

				if st.checkbox('Please click this checkbox after selecting continious features to scale'):
					X_train_scaled = scaler.fit_transform(X_train[selected_columns_to_scale])
					X_test_scaled = scaler.transform(X_test[selected_columns_to_scale])

					X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_columns_to_scale, index=X_train.index)
					X_train = pd.concat([X_train.drop(selected_columns_to_scale, axis=1), X_train_scaled], axis=1)
					st.write('Updated X_train set:')
					st.write('X_train shape: ', X_train.shape)
					st.dataframe(X_train)

					X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_columns_to_scale, index=X_test.index)
					X_test = pd.concat([X_test.drop(selected_columns_to_scale, axis=1), X_test_scaled], axis=1)
					st.write('Updated X_test set:')
					st.write('X_test shape: ', X_test.shape)
					st.dataframe(X_test)

						################################# MIN MAX SCALER ##############################

			if selected_scaler == 'Min-Max Scaler':
				scaler = MinMaxScaler()

				selected_columns_to_scale = st.multiselect('Please select continuous variable features to scale', X_train.columns)

				if st.checkbox('Please click this checkbox after selecting continious features to scale'):
					X_train_scaled = scaler.fit_transform(X_train[selected_columns_to_scale])
					X_test_scaled = scaler.transform(X_test[selected_columns_to_scale])

					X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_columns_to_scale, index=X_train.index)
					X_train = pd.concat([X_train.drop(selected_columns_to_scale, axis=1), X_train_scaled], axis=1)
					st.write('Updated X_train set:')
					st.write('X_train shape: ', X_train.shape)
					st.dataframe(X_train)

					X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_columns_to_scale, index=X_test.index)
					X_test = pd.concat([X_test.drop(selected_columns_to_scale, axis=1), X_test_scaled], axis=1)
					st.write('Updated X_test set:')
					st.write('X_test shape: ', X_test.shape)
					st.dataframe(X_test)

						################################### ROBUST SCALER ########################################


			if selected_scaler == 'Robust Scaler':
				scaler = RobustScaler()

				selected_columns_to_scale = st.multiselect('Please select continuous variable features to scale', X_train.columns)

				if st.checkbox('Please click this checkbox after selecting continious features to scale'):
					X_train_scaled = scaler.fit_transform(X_train[selected_columns_to_scale])
					X_test_scaled = scaler.transform(X_test[selected_columns_to_scale])

					X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_columns_to_scale, index=X_train.index)
					X_train = pd.concat([X_train.drop(selected_columns_to_scale, axis=1), X_train_scaled], axis=1)
					st.write('Updated X_train set:')
					st.write('X_train shape: ', X_train.shape)
					st.dataframe(X_train)

					X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_columns_to_scale, index=X_test.index)
					X_test = pd.concat([X_test.drop(selected_columns_to_scale, axis=1), X_test_scaled], axis=1)
					st.write('Updated X_test set:')
					st.write('X_test shape: ', X_test.shape)
					st.dataframe(X_test)
		st.write('_________________________________________________________________________________________________________________')

######################################################### FEATURE SELECTION ############################################################################

		st.subheader('Feature Selection')
		st.warning('WARNING: This is process intensive. Please carefully consider when using this feature')

		if st.checkbox('Need to decide which features to select and remove unimportant features?  Click the checkbox'):
			cv = RepeatedKFold(n_splits=5, n_repeats=25)


					############################## RFECV CLASSIFICATION #####################################

			if type_problem == 'Classification':
				scoring_type1_list = ['accuracy','balanced_accuracy','average_precision','precision','recall','roc_auc']
				scoring_type1 = st.selectbox('Select the scoring metric from the list',scoring_type1_list)
				if st.checkbox('Select the checkbox once a score metric is selected'):
					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def get_rfecv_classification(X_train1, y_train1):
						rfc = RandomForestClassifier()
						rfecv = RFECV(estimator=rfc, step=1, cv=cv, n_jobs=-1, scoring=scoring_type1)
						rfecv.fit(X_train1,y_train1)
						return rfecv
					rfecv = get_rfecv_classification(X_train, y_train)
					
					fig = plt.figure()
					plt.title('Number of Features verse {} score'.format(scoring_type1))
					plt.xlabel('Number of Features')
					plt.ylabel('{} score'.format(scoring_type1))
					plt.plot(range(1,len(rfecv.cv_results_['mean_test_score'])+1), rfecv.cv_results_['mean_test_score'])
					st.pyplot(fig)

					st.write('Number of features that are important: ', rfecv.n_features_)

					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def df_rfecv_get(rfecv):
						df_rfecv = pd.DataFrame(rfecv.feature_names_in_, columns=['Features'])
						df_rfecv['Importance'] = rfecv.support_
						df_rfecv = df_rfecv[df_rfecv['Importance']==True]
						df_rfecv['Coef'] = rfecv.estimator_.feature_importances_
						df_rfecv = df_rfecv.sort_values(by='Coef', ascending=False)
						return df_rfecv
					df_rfecv = df_rfecv_get(rfecv)

					st.write('Feature Importance')
					st.dataframe(df_rfecv)
					st.info('''NOTE: Make sure you understance the bias variance tradeoff.  Having too many features can overfit but having too few can underfit.  
						So, going after the highest score needs context when considering bias variance tradeoff.  More features can cause more variance.  
						Less features can cause more bias.''')
					


						############################### RFECV REGRESSION #######################################
				
			if type_problem == 'Regression':
				@st.cache(suppress_st_warning=True, allow_output_mutation=True)
				def get_rfecv_regression(X_train1, y_train1):
					rfr = RandomForestRegressor()
					rfecv = RFECV(estimator=rfr, step=1, cv=cv, n_jobs=-1, scoring='neg_root_mean_squared_error')
					rfecv.fit(X_train1, y_train1)
					return rfecv
				rfecv = get_rfecv_regression(X_train, y_train)
					
				fig = plt.figure()
				plt.title('Number of Features verse Accuracy Score')
				plt.xlabel('Number of Features')
				plt.ylabel('Accuracy Score')
				plt.plot(range(1,len(rfecv.cv_results_['mean_test_score'])+1), rfecv.cv_results_['mean_test_score'])
				st.pyplot(fig)

				st.write('Number of features that are important: ', rfecv.n_features_)

				@st.cache(suppress_st_warning=True, allow_output_mutation=True)
				def df_rfecv_get(rfecv):
					df_rfecv = pd.DataFrame(rfecv.feature_names_in_, columns=['Features'])
					df_rfecv['Importance'] = rfecv.support_
					df_rfecv = df_rfecv[df_rfecv['Importance']==True]
					df_rfecv['Coef'] = rfecv.estimator_.feature_importances_
					df_rfecv = df_rfecv.sort_values(by='Coef', ascending=False)
					return df_rfecv
				df_rfecv = df_rfecv_get(rfecv)

				st.write('Important Features:')
				st.dataframe(df_rfecv)
				st.write('Features in your current dataset not considered important:')
				st.dataframe(X_train.columns.drop(df_rfecv['Features'].unique()))
				st.info('''NOTE: Make sure you understance the bias variance tradeoff.  Having too many features can overfit but having too few can underfit.  
					So, going after the highest score needs context when considering bias variance tradeoff.  More features can cause more variance.  
					Less features can cause more bias.''') 

			st.write('_________________________________________________________________________________________________________________')
 
################################################################### SELECTING FEATURES #####################################

			st.subheader('Select Important Features')
			if st.checkbox('Click here to select features if you want to alter the features.  Otherwise, you can skip this step'):
				feature_options = ['Use Selected Features From Above', 'Choose the Features You Want']
				selected_feature_option = st.selectbox('Choose to use feature above or select the features you want:', feature_options)
				
				if selected_feature_option == 'Use Selected Features From Above':
					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def use_imp_features(X_train1, X_test1,df_rfecv1):
						importan_columns = df_rfecv1['Features'].unique()
						X_train2 = X_train1[importan_columns]
						X_test2 = X_test1[importan_columns]
						return X_train2, X_test2
					X_train, X_test = use_imp_features(X_train, X_test, df_rfecv)

					st.write('Updated X_train set:')
					st.write('X_train shape: ', X_train.shape)
					st.dataframe(X_train)

					st.write('Updated X_test set:')
					st.write('X_test shape', X_test.shape)
					st.dataframe(X_test)

				if selected_feature_option == 'Choose the Features You Want':
					importan_columns = st.multiselect('Select the features to include form the list', X_train.columns)
					if st.checkbox('Please check after selected features from the list above'):
						@st.cache(suppress_st_warning=True, allow_output_mutation=True)
						def use_sel_features(X_train1, X_test1, importan_columns1):
							X_train2 = X_train1[importan_columns1]
							X_test2 = X_test1[importan_columns1]
							return X_train2, X_test2
						X_train, X_test = use_sel_features(X_train, X_test, importan_columns)

						st.write('Updated X_train set:')
						st.write('X_train shape: ', X_train.shape)
						st.dataframe(X_train)

						st.write('Updated X_test set:')
						st.write('X_test shape', X_test.shape)
						st.dataframe(X_test)

		st.write('_________________________________________________________________________________________________________________')

#################################################################### PCA ###########################################################

		st.subheader('Dimensionality Reduction')
		if st.checkbox('Click here to perform PCA'):
			component_list = st.multiselect('Please select the number of principal components to test', range(1,30))
			component_list = [int(x) for x in component_list]
			cv1 = RepeatedKFold(n_splits=5, n_repeats=10)


			if st.checkbox('After selecting the number of principal components to test, check this box.  WARNING: If you need to change the component hyperparameters, then deselect this box first'):

				if type_problem == 'Regression':
					pipe = Pipeline([('pca', PCA()), ('rfr', RandomForestRegressor(n_jobs=-1))])
					params1 = {'pca__n_components':component_list}
					grid1 = GridSearchCV(estimator=pipe, param_grid=params1, n_jobs=-1, scoring='neg_root_mean_squared_error', cv=cv1)

					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def get_components(X_train1, y_train1,grid2):
						grid1.fit(X_train, y_train)
						return grid2
					grid1 = get_components(X_train,y_train,grid1)

					st.write('Best Neg RMSE: ', round(grid1.best_score_,3),'%')
					st.write('Best components: ', grid1.best_params_['pca__n_components'])


				if type_problem == 'Classification':
					pipe = Pipeline([('pca', PCA()), ('rfc', RandomForestClassifier(n_jobs=-1))])
					params1 = {'pca__n_components':component_list}
					grid1 = GridSearchCV(estimator=pipe, param_grid=params1, n_jobs=-1, scoring='accuracy', cv=cv1)

					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def get_components(X_train1, y_train1,grid2):
						grid2.fit(X_train1, y_train1)
						return grid2
					grid1 = get_components(X_train, y_train, grid1)

					st.write('Best Accuracy Score: ', round(grid1.best_score_*100,3),'%')
					st.write('Best components: ', grid1.best_params_['pca__n_components'])

				if st.checkbox('Check this box to apply the best components to the training and testing sets'):
					num_components = grid1.best_params_['pca__n_components']
					pca = PCA(n_components = num_components)

					X_train = pd.DataFrame(pca.fit_transform(X_train, y_train))
					X_test = pd.DataFrame(pca.transform(X_test))
					st.write('Updated X_train set:')
					st.dataframe(X_train)
					st.write('Updated X_test set:')
					st.dataframe(X_test)


		st.write('_________________________________________________________________________________________________________________')



########################################################################## MODEL ######################################################################

		st.subheader('The Model')



				########################################## CLASSIFICATION ################################

		if type_problem == 'Classification':

			classifier_list = ['RandomForestClassifier']
			classifier = st.selectbox('Select the classifier algorithm from the list:', classifier_list) 			

			st.info('NOTE: The next phase is Hyperparameter tuning.  It is assume there is some knowledge of hyperparameter tuning')

			if st.checkbox('Please select the checkbox when ready to run the tuning process'):

								################################## RANDOM FOREST CLASSIFIER ##################################

				if classifier == 'RandomForestClassifier':

					clf = RandomForestClassifier(n_jobs=-1)
					n_est = st.multiselect('Select the number of estimators from the list', range(25,500,25))
					n_est = [int(x) for x in n_est] # CONVERTING THE SELECTED VALUES FROM STR TO INT
					# st.write(n_est) # REMOVE LATER.  USING THIS FOR DEBUGGING
					scoring_type = st.selectbox('Please select the scoring metric from the list:', ['accuracy','balanced_accuracy','average_precision','precision','recall','roc_auc'])
					# st.write(scoring_type) # REMOVE LATER.  USING THIS FOR DEBUGGING

					cv = RepeatedKFold(n_splits=5, n_repeats=25)
					params = {'n_estimators': n_est}

					if st.checkbox('Select this checkbox after hyperparameters have been selected. WARNING: If you need to change Hyperparameters, then deselect this box first'):
						grid = GridSearchCV(estimator=clf, param_grid=params, n_jobs=-1, cv=cv, scoring=scoring_type)

						@st.cache(suppress_st_warning=True, allow_output_mutation=True)
						def get_hypertuned(X_train1,y_train1,grid1):
							grid1.fit(X_train1,y_train1)
							return grid1
						grid = get_hypertuned(X_train,y_train,grid)

						st.subheader('Model Cross Validation')
						st.info('NOTE: Repeated K-Fold is used for Cross Validation')
						st.write('Best Score {}:'.format(scoring_type), round(grid.best_score_*100,3),'%')
						st.write('Best Params: ', grid.best_params_['n_estimators'])
						st.warning('Change hyperparameters as needed to tune for the best scoring meteric selected.  Be sure to deselect the checkback above before making hyperparameter ajustments')
						st.info('After hyperparameter tuning is complete, move on to testing the model with the test data')

						st.write('_________________________________________________________________________________________________________________')


				################################### REGRESSION ####################################e


		if type_problem == 'Regression':

			regressor_list = ['RandomForestRegressor']
			regressor = st.selectbox('Select the regressor algorithm from the list:', regressor_list)

			st.info('NOTE: The next phase is Hyperparameter tuning.  It is assume there is some knowledge of hyperparameter tuning')

			if st.checkbox('Please select the checkbox when ready to run the tuning process'):

						################################## RANDOM FOREST REGRESSOR ##################################

				if regressor == 'RandomForestRegressor':
					rgr = RandomForestRegressor(n_jobs=-1)
					n_est = st.multiselect('Select the number of estimators from the list', range(25,500,25))
					n_est = [int(x) for x in n_est] # CONVERTING THE SELECTED VALUES FROM STR TO INT
					# st.write(n_est) # REMOVE LATER.  USING THIS FOR DEBUGGING
					scoring_type = st.selectbox('Please select the scoring metric from the list:', ['neg_root_mean_squared_error','r2'])
					# st.write(scoring_type) # REMOVE LATER.  USING THIS FOR DEBUGGING

					cv = RepeatedKFold(n_splits=5, n_repeats=25)
					params = {'n_estimators': n_est}

					if st.checkbox('Select this checkbox after hyperparameters have been selected. WARNING: If you need to change Hyperparameters, then deselect this box first'):
						grid = GridSearchCV(estimator=rgr, param_grid=params, n_jobs=-1, cv=cv, scoring=scoring_type)

						@st.cache(suppress_st_warning=True, allow_output_mutation=True)
						def get_hypertuned(X_train1,y_train1,grid1):
							grid1.fit(X_train1,y_train1)
							return grid1
						grid = get_hypertuned(X_train,y_train,grid)

						st.subheader('Model Cross Validation')
						st.info('NOTE: Repeated K-Fold is used for Cross Validation')
						st.write('Best Score {}:'.format(scoring_type), round(grid.best_score_,3),'%')
						st.write('Best Params: ', grid.best_params_['n_estimators'])
						st.warning('Change hyperparameters as needed to tune for the best scoring meteric selected.  Be sure to deselect the checkback above before making hyperparameter ajustments')
						st.info('After hyperparameter tuning is complete, move on to testing the model with the test data')

						st.write('_________________________________________________________________________________________________________________')


############################################### MODEL TESTING ###############################################################

		st.subheader('Model Testing')

		if st.checkbox('WARNING: ONLY CHECK THIS BOX IF YOU ARE TRUELY READY TO TEST THE MODEL'):
			
			if type_problem == 'Regression':
				score = grid.score(X_test,y_test)
				st.write('Model Score {}:'.format(scoring_type), round(score,3),'%')

				y_pred = pd.DataFrame(grid.predict(X_test), columns=['Predictions'], index=y_test.index)
				df_compare = pd.concat([y_test, y_pred], axis=1)
				df_compare['Abs Difference'] = np.ravel(y_test) - np.ravel(y_pred)
				st.write('Actual Target Values verse Predicted Target Values:')
				st.dataframe(df_compare)

				st.write('Scatter Plot Resizing Options:')
				height0 = st.selectbox('Resize height', range(6,32,2))
				width0 = st.selectbox('Resize width', range(6,32,2))

				fig0 = plt.figure(figsize=(width0,height0))
				sns.scatterplot(x=np.ravel(y_pred),y=np.ravel(y_test))
				plt.xlabel('Predicted Target Values')
				plt.ylabel('Actual Target Values')
				plt.title('Actual Target Values verse Predicted Target Values')
				st.pyplot(fig0)



			if type_problem == 'Classification':
				score = grid.score(X_test,y_test)
				st.write('Model Score {}:'.format(scoring_type), round(score*100,3),'%')

				y_pred = pd.DataFrame(grid.predict(X_test), columns=['Predictions'], index=y_test.index)
				df_compare = pd.concat([y_test, y_pred], axis=1)
				st.write('Actual Target Values verse Predicted Target Values:')
				st.dataframe(df_compare)

				cm = confusion_matrix(y_test,y_pred)

				st.write('Confusion Matrix Resizing Options')
				height9 = st.selectbox('Resize height', range(4,32,2))
				width9 = st.selectbox('Resize width', range(4,32,2))
				
				fig9 = plt.figure(figsize=(width9,height9))
				sns.heatmap(cm, cmap='Blues', annot=True, cbar=False, linewidths=1, linecolor='black')
				plt.xlabel('Predicted Target Values')
				plt.ylabel('True Target Values')
				plt.title('Confusion Matrix')
				st.pyplot(fig9)


############################################### CLOSING #####################################################
		st.write('_________________________________________________________________________________________________________________')
		st.write('')
		st.header('Thank you for stopping by and checking out my machine learning web app')

if __name__ == '__main__':
	main()