import pandas as pd
import pickle

print('Creating dataset with dummy variables...')

headers = ['CASE_STATE', 'AGE', 'SEX', 'PERSON_TYPE', 'SEATING_POSITION', 'RESTRAINT_SYSTEM-USE',
           'AIR_BAG_AVAILABILITY_OR_DEPLOYMENT', 'EJECTION', 'EJECTION_PATH', 'EXTRICATION',
           'NON_MOTORIST_LOCATION', 'POLICE_REPORTED_ALCOHOL_INVOLVEMENT', 'METHOD_ALCOHOL_DETERMINATION',
           'ALCOHOL_TEST_TYPE', 'ALCOHOL_TEST_RESULT', 'POLICE-REPORTED_DRUG_INVOLVEMENT',
           'METHOD_OF_DRUG_DETERMINATION', 'DRUG_TEST_TYPE', 'DRUG_TEST_RESULTS_(1_of_3)',
           'DRUG_TEST_TYPE_(2_of_3)', 'DRUG_TEST_RESULTS_(2_of_3)', 'DRUG_TEST_TYPE_(3_of_3)',
           'DRUG_TEST_RESULTS_(3_of_3)', 'HISPANIC_ORIGIN', 'TAKEN_TO_HOSPITAL',
           'RELATED_FACTOR_(1)-PERSON_LEVEL', 'RELATED_FACTOR_(2)-PERSON_LEVEL',
           'RELATED_FACTOR_(3)-PERSON_LEVEL', 'RACE', 'INJURY_SEVERITY']

# Import database as pandas DataFrame
df_train = pd.read_csv('data/fars_train.csv', header=None, names=headers)
df_test = pd.read_csv('data/fars_test.csv', header=None, names=headers)


train_length = len(df_train)
df = pd.concat([df_train, df_test], axis=0)
df = df.applymap(str)

# Formating quantitative variables to int so that they are not transformed to dummy variables
df['AGE'] = df['AGE'].astype('int32')
df['ALCOHOL_TEST_RESULT'] = df['ALCOHOL_TEST_RESULT'].astype('int32')
df['DRUG_TEST_RESULTS_(1_of_3)'] = df['DRUG_TEST_RESULTS_(1_of_3)'].astype('int32')
df['DRUG_TEST_RESULTS_(2_of_3)'] = df['DRUG_TEST_RESULTS_(2_of_3)'].astype('int32')
df['DRUG_TEST_RESULTS_(3_of_3)'] = df['DRUG_TEST_RESULTS_(3_of_3)'].astype('int32')

# Converting string type variable into dummies
df_to_dummy = pd.get_dummies(df)
df_to_dummy = df_to_dummy.drop(['INJURY_SEVERITY_nan'], axis=1)

# Spliting back train/test sets
train_set = df_to_dummy.iloc[:train_length, :-8]
test_set = df_to_dummy.iloc[train_length:, :-8]

# Creating multimodal label into a signle variable
df_labels = pd.DataFrame(df_train.iloc[:, -1])
df_labels['INJURY_SEVERITY'] = df_labels['INJURY_SEVERITY'].astype('category')
df_labels_categories = df_labels['INJURY_SEVERITY'].cat.codes

# Creating a label/index dictionary to decode the classification output
label_dict = {}
for i in range(len(df_labels)):
    if df_labels.iloc[i] not in label_dict:
        label_dict[df_labels.iloc[i]] = df_labels_categories.iloc[i]

# Dumping dataset into a .pkl file
with open('data/dataset_dummy.pkl', 'wb') as f:
    pickle.dump([train_set, test_set, df_labels_categories, label_dict], f)

print('Done')
print('-' * 50)

print('Creating dataset with multimodal variables...')

df_train = pd.read_csv('data/fars_train.csv', header=None, names=headers)
train_length = len(df_train)
train_data = df_train.iloc[:, :-1]
df_labels = df_train.iloc[:, -1]

df_test = pd.read_csv('data/fars_test.csv', header=None, names=headers)
test_data = df_test.iloc[:, :-1]

df = pd.concat([train_data, test_data], axis=0)
df_labels = pd.DataFrame(df_labels)

column_names = list(df.columns.values)
# We suppress quantitative variables which we donnot want to convert into indexes
column_names.remove('AGE')
column_names.remove('ALCOHOL_TEST_RESULT')
column_names.remove('DRUG_TEST_RESULTS_(1_of_3)')
column_names.remove('DRUG_TEST_RESULTS_(2_of_3)')
column_names.remove('DRUG_TEST_RESULTS_(3_of_3)')

# Converting each category variables to index for each qualitative variable
for column_name in column_names:
    df[column_name] = df[column_name].astype('category')
    df[column_name] = df[column_name].cat.codes

# Spliting back to train/test
train_data = df.iloc[:train_length, :]
test_data = df.iloc[train_length:, :]

# Dumping dataset into a .pkl file
with open('data/dataset_multimodal.pkl', 'wb') as f:
    pickle.dump([train_data, test_data, df_labels_categories, label_dict], f)

print('Done')

print('=' * 50)
print('END OF DATA PROCESSING')
print('=' * 50)
