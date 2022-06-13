from azureml.core import Workspace
from azureml.core import Datastore, Dataset
import datetime as dt
import pandas as pd
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.datadrift import DataDriftDetector
from azureml.widgets import RunDetails

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Upload the baseline data
default_ds = ws.get_default_datastore()
default_ds.upload_files(files=['./data/pre_processed.csv', './data/pre_processed1.csv'],
                       target_path='loan-baseline',
                       overwrite=True, 
                       show_progress=True)

# Create and register the baseline dataset
print('Registering baseline dataset...')
baseline_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'loan-baseline/*.csv'))
baseline_data_set = baseline_data_set.register(workspace=ws, 
                           name='loan baseline',
                           description='loan baseline data',
                           tags = {'format':'CSV'},
                           create_new_version=True)

# Load the smaller of the two data files
data = pd.read_csv('data/pre_processed1.csv')

# We'll generate data for the past 6 weeks
weeknos = reversed(range(6))

file_paths = []
for weekno in weeknos:
    
    # Get the date X weeks ago
    data_date = dt.date.today() - dt.timedelta(weeks=weekno)
    
    # Modify data to ceate some drift
    data['Experience'] = round(data['Experience'] * 1.3).astype(int)
    data['Income'] = round(data['Income'] * 1.3).astype(int)
    data['Age'] = round(data['Age'] * 1.2).astype(int)
    data['CCAvg'] = round(data['CCAvg'] * 1.3).astype(int)
    data['Mortgage'] = round(data['Mortgage'] * 1.3).astype(int)
    
    # Save the file with the date encoded in the filename
    file_path = 'data/loan_{}.csv'.format(data_date.strftime("%Y-%m-%d"))
    data.to_csv(file_path)
    file_paths.append(file_path)

# Upload the files
path_on_datastore = 'loan-target'
default_ds.upload_files(files=file_paths,
                       target_path=path_on_datastore,
                       overwrite=True,
                       show_progress=True)

# Use the folder partition format to define a dataset with a 'date' timestamp column
partition_format = path_on_datastore + '/loan_{date:yyyy-MM-dd}.csv'
target_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, path_on_datastore + '/*.csv'),
                                                       partition_format=partition_format)

# Register the target dataset
#print('Registering target dataset...')
target_data_set = target_data_set.with_timestamp_columns('date').register(workspace=ws,
                                                                          name='loan target',
                                                                          description='loan target data',
                                                                          tags = {'format':'CSV'},
                                                                          create_new_version=True)

#print('Target dataset registered!')  


cluster_name = "your-compute-cluster"

try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)


# set up feature list
features = features = ['Age', 'Experience', 'Income','CCAvg','Mortgage']
# set up data drift detector
monitor = DataDriftDetector.create_from_datasets(ws, 'mslearn-loan-drift', baseline_data_set, target_data_set,
                                                      compute_target=cluster_name, 
                                                      frequency='Week', 
                                                      feature_list=features, 
                                                      drift_threshold=.3, 
                                                      latency=24)

backfill = monitor.backfill(dt.datetime.now() - dt.timedelta(weeks=6), dt.datetime.now())
RunDetails(backfill).show()
backfill.wait_for_completion()

drift_metrics = backfill.get_metrics()
for metric in drift_metrics:
    print(metric, drift_metrics[metric])
