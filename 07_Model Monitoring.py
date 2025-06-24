#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Code Block 1 (Part B Imports and SageMaker Session Setup)
import pandas as pd
import boto3
import sagemaker
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import os
import numpy as np # Make sure numpy is imported

# --- SageMaker Session Setup (ESSENTIAL FOR ALL NOTEBOOK OPERATIONS) ---
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sagemaker_session.boto_region_name
bucket = sagemaker_session.default_bucket() # Your default S3 bucket
prefix = "crime-data-ml-project" # A new prefix for this project's S3 data/models

print(f"SageMaker session initialized in region: {region}")
print(f"Default S3 bucket: {bucket}")
print(f"IAM role ARN: {role}")


# In[3]:


# S3 File Accessibility Check (FIXED object_key)
import boto3

# Ensure 'region' is defined from your initial setup (Code Block 1)
s3_client_check = boto3.client('s3', region_name=region)

bucket_name = 'sagemaker-us-east-1-763564741945'
# FIXED: Key is now just the filename as it's in the root
object_key = 'crime_data_10k_rows.csv' # <--- FIXED OBJECT KEY!

print(f"Attempting to check S3 object: s3://{bucket_name}/{object_key}")

try:
    response = s3_client_check.head_object(Bucket=bucket_name, Key=object_key)
    print("\nS3 Object Check SUCCESSFUL!")
    print("File exists and is accessible. Details:")
    print(f"  Content-Length: {response.get('ContentLength')} bytes")
    print(f"  Content-Type: {response.get('ContentType')}")
    print(f"  LastModified: {response.get('LastModified')}")
    file_accessible = True
except s3_client_check.exceptions.ClientError as e:
    error_code = int(e.response['Error']['Code'])
    if error_code == 404:
        print(f"\nS3 Object Check FAILED: File Not Found (404 Error).")
        print(f"Please double-check the bucket name '{bucket_name}' and object key '{object_key}'.")
        print(f"Verify in the AWS S3 console that the file exists at this EXACT path.")
    elif error_code == 403:
        print(f"\nS3 Object Check FAILED: Access Denied (403 Error).")
        print(f"Your SageMaker notebook's IAM role (LabRole) might not have 's3:GetObject' permissions for this file.")
    else:
        print(f"\nS3 Object Check FAILED: An unexpected error occurred: {e}")
    file_accessible = False
except Exception as e:
    print(f"\nAn unexpected Python error occurred during S3 check: {e}")
    file_accessible = False

if not file_accessible:
    print("\nCannot proceed with pd.read_csv if file is not accessible via boto3.")
    print("Please resolve the S3 access issue first.")


# In[4]:


# Code Block 2 (Part B Data Loading - from S3 - FINAL REFRESH)

s3_crime_data_path = 's3://sagemaker-us-east-1-763564741945/crime_data_10k_rows.csv' # <--- VERIFIED S3 URI!

try:
    df_raw_limited = pd.read_csv(s3_crime_data_path, encoding='latin-1')
    print(f"DataFrame shape after loading from S3: {df_raw_limited.shape}")
    display(df_raw_limited.head()) # This will show the first few rows
except Exception as e:
    print(f"Error loading the dataset from S3: {e}")
    df_raw_limited = None

df_processed = df_raw_limited.copy() # This will now work as df_raw_limited is not None


# In[4]:


# Part B: Data Preprocessing (Full Code Blocks 3-6)
# Code Block 3 (Part B Initial Data Inspection on df_processed)

print("DataFrame Shape after loading essential columns:", df_processed.shape)
print("\nDataFrame Info after loading essential columns:")
df_processed.info()

print("\nColumn Names after loading essential columns:")
print(df_processed.columns.tolist())

print("\nUnique values in 'Part 1-2':")
print(df_processed['Part 1-2'].value_counts())

print("\nUnique values in 'AREA NAME' (first 10, if many):")
print(df_processed['AREA NAME'].value_counts().head(10))

print("\nUnique values in 'Crm Cd Desc' (first 10, if many):")
print(df_processed['Crm Cd Desc'].value_counts().head(10))


# In[5]:


# Code Block 4 (Part B Data Preprocessing - Initial Cleaning & Target Prep)

# Make a copy if df_processed was modified in previous cell, otherwise it's already a copy from initial load
# df_processed = df_raw_limited.copy() # This line is already in the REVISED Code Block 2, so comment out or ensure not duplicated if needed.

# --- 1. Target Variable Transformation (Part 1-2: change 2 to 0) ---
df_processed['Part 1-2'] = df_processed['Part 1-2'].replace({2: 0})
print("Transformed 'Part 1-2' value counts:")
print(df_processed['Part 1-2'].value_counts())

# --- 2. Handle Dates and Extract Features ---
# Convert 'DATE OCC' to datetime objects
# UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`.
# To avoid this, specify format: df_processed['DATE OCC'] = pd.to_datetime(df_processed['DATE OCC'], format='%m/%d/%Y')
df_processed['DATE OCC'] = pd.to_datetime(df_processed['DATE OCC'])
df_processed['OCC_HOUR'] = df_processed['TIME OCC'].astype(int) # Assuming TIME OCC is already hour (0-23)
df_processed['OCC_DAY_OF_WEEK'] = df_processed['DATE OCC'].dt.dayofweek
df_processed['OCC_MONTH'] = df_processed['DATE OCC'].dt.month
df_processed['OCC_YEAR'] = df_processed['DATE OCC'].dt.year

# Drop original date and time columns after extraction
df_processed = df_processed.drop(columns=['DATE OCC', 'TIME OCC']) # 'Date Rptd' was not loaded via usecols
print(f"\nDataFrame shape after date engineering: {df_processed.shape}")

# --- 3. Handle Missing Values in remaining Victim/Premise columns (Impute with Mode) ---
# Vict Sex, Vict Descent, Premis Cd, Premis Desc
for col in ['Vict Sex', 'Vict Descent', 'Premis Cd', 'Premis Desc']:
    if col in df_processed.columns:
        if df_processed[col].isnull().any():
            mode_val = df_processed[col].mode()[0]
            df_processed[col] = df_processed[col].fillna(mode_val)
            print(f"Filled missing values in '{col}' with mode: {mode_val}")
        else:
            print(f"No missing values in '{col}'.")
    else:
        print(f"Column '{col}' not found (already excluded by usecols or named differently).")

# --- Verify no more missing values in processed columns ---
print("\nMissing values after imputation:")
print(df_processed[['Vict Sex', 'Vict Descent', 'Premis Cd', 'Premis Desc']].isnull().sum())

print("\nDataFrame Info after initial processing:")
df_processed.info(verbose=True, show_counts=True)


# In[6]:


# Code Block 5 (Part B - Check Cardinality of Categorical Features)

# Identify remaining object columns (potential categoricals)
object_cols = df_processed.select_dtypes(include='object').columns

print("\nUnique value counts for remaining object columns:")
for col in object_cols:
    print(f"\n--- Column: '{col}' ({df_processed[col].nunique()} unique values) ---")
    if df_processed[col].nunique() < 50:
        print(df_processed[col].value_counts())
    else:
        print(df_processed[col].value_counts().head(10))
        print(f"    ... {df_processed[col].nunique() - 10} more unique values")


# In[8]:


# Code Block 6 (REVISED WORKFLOW)

# Make a copy for final feature set
df_final = df_processed.copy()

# --- 1. Clean up 'Vict Sex' and 'Vict Descent' (Your code here is perfect) ---
df_final['Vict Sex'] = df_final['Vict Sex'].replace(['X', 'H', '-'], 'UNKNOWN')
top_descents = df_final['Vict Descent'].value_counts().nlargest(10).index.tolist()
df_final['Vict Descent'] = df_final['Vict Descent'].apply(
    lambda x: x if x in top_descents else 'UNKNOWN'
)

# --- 2. Define and Drop Unnecessary Columns (CRITICAL NEW STEP) ---
# These are identifiers, redundant fields, or high-cardinality text we won't use.
cols_to_drop = [
    'DR_NO',
    'Rpt Dist No',
    'Part 1-2', # Drop the target from the feature set
    'Crm Cd', 'Crm Cd Desc', # High cardinality / redundant
    'Mocodes', # High cardinality text
    'Premis Cd', # Redundant with Premis Desc
    'Weapon Used Cd', 'Weapon Desc', # Too many missing values
    'Status', 'Status Desc', # Leakage: This describes the outcome of the crime
    'Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', # Mostly null or redundant
    'LOCATION', 'Cross Street', # Raw location data
    'Date Rptd', # This column was not in the previous df.info(), but if it exists, drop it.
]

# Drop columns that exist in the DataFrame, ignore errors for any that don't
df_final = df_final.drop(columns=cols_to_drop, errors='ignore')


# --- 3. Separate Features (X) and Target (y) ---
# The target variable 'Part 1-2' is in df_processed
y = df_processed['Part 1-2']
X = df_final

print(f"Shape of features (X) before encoding: {X.shape}")


# --- 4. Perform One-Hot Encoding on the Feature Set (X) ---
# Identify remaining categorical columns in our feature set X
categorical_cols_in_X = X.select_dtypes(include='object').columns.tolist()

print(f"\nColumns to be one-hot encoded: {categorical_cols_in_X}")

X = pd.get_dummies(X, columns=categorical_cols_in_X, drop_first=True, dtype=int)

print(f"Shape of features (X) after encoding: {X.shape}")


# --- 5. Display Final Results ---
print(f"\nFinal shape of features (X): {X.shape}")
print(f"Final shape of target (y): {y.shape}")
print("\nFinal feature data types and non-null counts:")
X.info(verbose=True, show_counts=True)


# In[9]:


from sklearn.model_selection import train_test_split
import pandas as pd # Ensure pandas is imported if not already

# Convert X and y to numpy arrays.
X_np = X.values
y_np = y.values

# --- First Split: Separate Production Data (40%) ---
# Remaining data (X_temp_for_t_v_t, y_temp_for_t_v_t) will be 60% of original
X_temp_for_t_v_t, X_prod, y_temp_for_t_v_t, y_prod = train_test_split(
    X_np, y_np, test_size=0.4, random_state=42, stratify=y_np
)

# --- Second Split: Divide the remaining 60% (X_temp_for_t_v_t) into Train (40%), Val (10%), Test (10%) ---
# Split X_temp_for_t_v_t (60% of original) into actual train (40% of original) and temp (20% of original)
X_train, X_temp_val_test, y_train, y_temp_val_test = train_test_split(
    X_temp_for_t_v_t, y_temp_for_t_v_t, test_size=(1/3), random_state=42, stratify=y_temp_for_t_v_t # <--- FIX HERE!
)

# Split X_temp_val_test (20% of original) into Validation (10%) and Test (10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp_val_test, y_temp_val_test, test_size=0.5, random_state=42, stratify=y_temp_val_test
)


print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"X_prod shape: {X_prod.shape}, y_prod shape: {y_prod.shape}")

# Confirm total rows
total_rows = X_train.shape[0] + X_val.shape[0] + X_test.shape[0] + X_prod.shape[0]
print(f"Total rows after split: {total_rows} (should be {X_np.shape[0]})")

# Confirm distribution in splits
print("\nTarget distribution in y_train:")
print(pd.Series(y_train).value_counts(normalize=True))
print("\nTarget distribution in y_val:")
print(pd.Series(y_val).value_counts(normalize=True))
print("\nTarget distribution in y_test:")
print(pd.Series(y_test).value_counts(normalize=True))
print("\nTarget distribution in y_prod:")
print(pd.Series(y_prod).value_counts(normalize=True))


# In[10]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np # Ensure numpy is imported if not already
import pandas as pd # Ensure pandas is imported if not already

# Find the most frequent class in the training data
majority_class = pd.Series(y_train).mode()[0]
print(f"Majority class in training data: {majority_class}")

# Predict the majority class for all instances in the test set
y_pred_benchmark = np.full_like(y_test, fill_value=majority_class)

# Evaluate the benchmark model on the test set
benchmark_accuracy = accuracy_score(y_test, y_pred_benchmark)
benchmark_precision = precision_score(y_test, y_pred_benchmark, zero_division=0) # zero_division=0 to handle cases where no positive predictions are made
benchmark_recall = recall_score(y_test, y_pred_benchmark, zero_division=0)
benchmark_f1 = f1_score(y_test, y_pred_benchmark, zero_division=0)
benchmark_cm = confusion_matrix(y_test, y_pred_benchmark)


print("\n--- Benchmark Model Evaluation (Majority Class Predictor) ---")
print(f"Accuracy: {benchmark_accuracy:.4f}")
print(f"Precision: {benchmark_precision:.4f}")
print(f"Recall: {benchmark_recall:.4f}")
print(f"F1-Score: {benchmark_f1:.4f}")
print("Confusion Matrix:\n", benchmark_cm)

# Store benchmark results for later comparison
benchmark_results = {
    "model": "Majority Class Predictor",
    "accuracy": benchmark_accuracy,
    "precision": benchmark_precision,
    "recall": benchmark_recall,
    "f1_score": benchmark_f1
}


# In[12]:


import pandas as pd # Ensure pandas is imported
import os # Ensure os is imported for path handling

# Combine target and features for training and validation sets
train_df = pd.DataFrame(y_train, columns=['target'])
train_df = pd.concat([train_df, pd.DataFrame(X_train)], axis=1)

val_df = pd.DataFrame(y_val, columns=['target'])
val_df = pd.concat([val_df, pd.DataFrame(X_val)], axis=1)

print("Train DataFrame head (target should be first column):")
print(train_df.head())
print(f"\nTrain DataFrame shape: {train_df.shape}")
print(f"Validation DataFrame shape: {val_df.shape}")

# Define local file paths for CSVs
local_train_path = 'train.csv'
local_val_path = 'validation.csv'

# Save DataFrames to local CSV files
train_df.to_csv(local_train_path, header=False, index=False)
val_df.to_csv(local_val_path, header=False, index=False)
print(f"\nSaved training data to local file: {local_train_path}")
print(f"Saved validation data to local file: {local_val_path}")


# Upload local CSV files to S3
# Ensure 'bucket' and 'sagemaker_session' are defined from Code Block 1
# And 'prefix' is defined (e.g., prefix = "crime-data-ml-project")
train_s3_uri = sagemaker_session.upload_data(
    path=local_train_path,  # <--- Pass the local file path here
    bucket=bucket,
    key_prefix=f"{prefix}/train" # Ensure this path is correct
)

val_s3_uri = sagemaker_session.upload_data(
    path=local_val_path,    # <--- Pass the local file path here
    bucket=bucket,
    key_prefix=f"{prefix}/validation" # Ensure this path is correct
)

print(f"\nTraining data uploaded to S3: {train_s3_uri}")
print(f"Validation data uploaded to S3: {val_s3_uri}")

# Define Sagemaker TrainingInput objects
from sagemaker.inputs import TrainingInput

train_input = TrainingInput(train_s3_uri, content_type='text/csv')
val_input = TrainingInput(val_s3_uri, content_type='text/csv')


# In[14]:


# 1. Define the local path for the batch input file
local_batch_input_path = 'test_batch_data.csv'

# 2. Save ONLY the features (X_test) to a CSV
# Make sure X_test is a pandas DataFrame or NumPy array with 208 columns
pd.DataFrame(X_test).to_csv(local_batch_input_path, header=False, index=False)

print(f"Saved batch input data locally to: {local_batch_input_path}")

# 3. Upload this new file to S3
batch_input_s3_uri = sagemaker_session.upload_data(
    path=local_batch_input_path,
    bucket=bucket,
    key_prefix=f"{prefix}/batch-input"
)

print(f"Batch input data uploaded to S3: {batch_input_s3_uri}")

# 4. Now, create your transformer and run .transform() using this new URI
# transformer = xgb_estimator.transformer(...)
# transformer.transform(data=batch_input_s3_uri, ...)


# In[15]:


from sagemaker.amazon.amazon_estimator import image_uris
from time import gmtime, strftime

# Get the XGBoost image URI for your region
# Ensure 'region' is defined from Code Block 1
xgb_container_image = image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1" # Using a stable version, e.g., 1.7-1 or 1.9-1 (check latest available)
)
print(f"XGBoost container image: {xgb_container_image}")

# Define a unique training job name
training_job_name = f"crime-prediction-xgb-{strftime('%Y%m%d-%H%M%S', gmtime())}"
output_s3_path = f"s3://{bucket}/{prefix}/output" # S3 path for model artifacts

# Create a SageMaker Estimator for XGBoost
# Ensure 'role', 'sagemaker_session' are defined from Code Block 1
xgb_estimator = sagemaker.estimator.Estimator(
    image_uri=xgb_container_image,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge", # Using a sufficiently powerful instance
    volume_size=20, # GB, for data storage on instance
    output_path=output_s3_path,
    sagemaker_session=sagemaker_session
)

# Set hyperparameters for XGBoost (good starting point for binary classification)
xgb_estimator.set_hyperparameters(
    objective="binary:logistic", # For binary classification
    eval_metric="auc", # Evaluation metric during training (Area Under the Curve)
    num_round=100, # Number of boosting rounds (iterations)
    eta=0.2, # Learning rate
    max_depth=5, # Max depth of a tree
    subsample=0.7, # Subsample ratio of the training instance
    colsample_bytree=0.7, # Subsample ratio of columns when constructing each tree
    gamma=0.1 # Minimum loss reduction required to make a further partition
)

# Define data channels for training and validation
data_channels = {
    'train': train_input, # train_input from Code Block 9
    'validation': val_input # val_input from Code Block 9
}

print(f"\nStarting XGBoost training job: {training_job_name}...")
# Start the training job
xgb_estimator.fit(inputs=data_channels, job_name=training_job_name, logs=True)

print(f"\nTraining job '{training_job_name}' completed. Model artifacts saved to: {xgb_estimator.model_data}")


# In[16]:


from sagemaker.transformer import Transformer
from time import gmtime, strftime
from sagemaker.model import Model # New Import

# Define a unique name for the batch transform job
transform_job_name = f"crime-prediction-batch-{strftime('%Y%m%d-%H%M%S', gmtime())}"

# The name of the model in SageMaker will be the training job name
model_name_from_training = xgb_estimator.latest_training_job.job_name

# --- Explicitly create and register the SageMaker Model resource ---
print(f"Attempting to create SageMaker Model resource: {model_name_from_training}...")
model_obj = Model(
    image_uri=xgb_estimator.image_uri, # Use the image URI from the estimator
    model_data=xgb_estimator.model_data, # Use the model data S3 URI from the estimator
    role=role, # Your IAM role
    sagemaker_session=sagemaker_session,
    name=model_name_from_training # Explicitly set the name for the Model resource
)

try:
    # This call creates the Model resource in SageMaker's backend
    model_obj.create()
    # FIX: Removed .arn from print statement as it might not be immediately available
    print(f"SageMaker Model resource '{model_obj.name}' created successfully.") # <--- FIX HERE
except Exception as e:
    # Handle case where model might already exist from a previous run if name is not unique enough
    if "Model with name " in str(e) and "already exists" in str(e):
        print(f"Model resource '{model_obj.name}' already exists. Proceeding.")
    else:
        raise e # Re-raise other unexpected errors


# Create a Transformer object using the explicitly created model_obj.name
transformer = Transformer(
    model_name=model_obj.name, # Use the name from the explicitly created Model object
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket}/{prefix}/batch-transform-output/{transform_job_name}",
    sagemaker_session=sagemaker_session,
    base_transform_job_name="crime-prediction-batch"
)
print(f"Batch Transformer configured. Model name: {transformer.model_name}")


# Convert X_test to a DataFrame and upload to S3 for batch transform
test_df_for_batch = pd.DataFrame(X_test)
test_batch_local_path = 'test_batch_data.csv'
test_df_for_batch.to_csv(test_batch_local_path, header=False, index=False)

test_batch_s3_uri = sagemaker_session.upload_data(
    path=test_batch_local_path,
    bucket=bucket,
    key_prefix=f"{prefix}/batch-input"
)
print(f"Test batch data uploaded to S3: {test_batch_s3_uri}")

print(f"\nStarting batch transform job: {transform_job_name}...")
# Start the batch transform job
transformer.transform(
    data=test_batch_s3_uri,
    content_type="text/csv",
    split_type="Line",
    job_name=transform_job_name
)

# Wait for the batch transform job to complete
print("Waiting for batch transform job to complete...")
transformer.wait()
print(f"Batch transform job '{transform_job_name}' completed.")

# Store the S3 output path of the batch transform
batch_output_s3_uri = transformer.output_path
print(f"Batch predictions saved to: {batch_output_s3_uri}")


# In[18]:


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from urllib.parse import urlparse

# --- 1. Download Predictions from S3 (Corrected) ---
# Get the base output path directly from the transformer object
predictions_s3_uri = transformer.output_path

# The output filename is the name of your input file with '.out' appended
# We get the input filename from the URI we used for the transform job
input_filename = os.path.basename(batch_input_s3_uri) 
output_filename = input_filename + '.out' # e.g., 'test_batch_data.csv.out'
local_predictions_path = 'test_predictions.csv'

print(f"Downloading predictions from: {os.path.join(predictions_s3_uri, output_filename)}")

# Parse the S3 URI to get the bucket and key
parsed_s3_uri = urlparse(predictions_s3_uri)
s3_bucket = parsed_s3_uri.netloc
# The key is the path part, plus the output filename
s3_key_prefix_for_download = os.path.join(parsed_s3_uri.path.lstrip('/'), output_filename)

# Download the specific output file
sagemaker_session.download_data(
    path='.', # Download to the current directory
    bucket=s3_bucket,
    key_prefix=s3_key_prefix_for_download
)

# Rename the downloaded file for clarity
os.rename(output_filename, local_predictions_path)
print(f"Predictions downloaded and saved locally to: {local_predictions_path}")


# --- 2. Load Predictions and Convert to Classes (This part is unchanged) ---
y_pred_probs = pd.read_csv(local_predictions_path, header=None).values.flatten()
y_pred_xgb = (y_pred_probs >= 0.5).astype(int)


# --- 3. Evaluate the XGBoost Model (This part is unchanged) ---
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_cm = confusion_matrix(y_test, y_pred_xgb)

print("\n--- XGBoost Model Evaluation ---")
print(f"Accuracy: {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall: {xgb_recall:.4f}")
print(f"F1-Score: {xgb_f1:.4f}")
print("Confusion Matrix:\n", xgb_cm)

print("\n--- Comparison with Benchmark ---")
print(f"Benchmark F1-Score: {benchmark_results['f1_score']:.4f}")
print(f"XGBoost F1-Score:   {xgb_f1:.4f}")

if xgb_f1 > benchmark_results['f1_score']:
    print("\n✅ XGBoost model significantly outperforms the benchmark.")
else:
    print("\n⚠️ XGBoost model does not outperform the benchmark. Hyperparameter tuning may be needed.")


# In[19]:


# ======================================================
# Part 3: Model Monitoring and Bias Detection
# ======================================================
## Applying Lab 5.1 Code


# In[25]:


get_ipython().run_cell_magic('time', '', '\nfrom datetime import datetime, timedelta, timezone\nimport json\nimport os\nimport re\nimport boto3\nfrom time import sleep\nfrom threading import Thread\n\nimport pandas as pd\n\nfrom sagemaker import get_execution_role, session, Session, image_uris\nfrom sagemaker.s3 import S3Downloader, S3Uploader\nfrom sagemaker.processing import ProcessingJob\nfrom sagemaker.serializers import CSVSerializer\n\nfrom sagemaker.model import Model\nfrom sagemaker.model_monitor import DataCaptureConfig\n\nsession = Session()\n')


# In[26]:


# Get Execution role
role = get_execution_role()
print("RoleArn:", role)

region = session.boto_region_name
print("Region:", region)


# In[27]:


# You can use a different bucket, but make sure the role you chose for this notebook
# has the s3:PutObject permissions. This is the bucket into which the data is captured
bucket = session.default_bucket()
print("Demo Bucket:", bucket)

# ==> CHANGE THIS LINE <==
# Use the prefix from your own project for better S3 organization
prefix = "crime-data-ml-project" 

##S3 prefixes
data_capture_prefix = f"{prefix}/datacapture"
s3_capture_upload_path = f"s3://{bucket}/{data_capture_prefix}"

ground_truth_upload_path = (
    f"s3://{bucket}/{prefix}/ground_truth_data/{datetime.now():%Y-%m-%d-%H-%M-%S}"
)

reports_prefix = f"{prefix}/reports"
s3_report_path = f"s3://{bucket}/{reports_prefix}"

##Get the model monitor image
monitor_image_uri = image_uris.retrieve(framework="model-monitor", region=region)

print("Image URI:", monitor_image_uri)
print(f"Capture path: {s3_capture_upload_path}")
print(f"Ground truth path: {ground_truth_upload_path}")
print(f"Report path: {s3_report_path}")


# In[28]:


#2.3 Deploy the model with data capture enabled


# In[30]:


# Use a name that reflects your project
endpoint_name = f"crime-prediction-xgb-monitor-{datetime.utcnow():%Y-%m-%d-%H%M}"
print("EndpointName =", endpoint_name)

# This configures the endpoint to log all incoming and outgoing data to S3
# This is the data our monitors will analyze.
data_capture_config = DataCaptureConfig(
    enable_capture=True, 
    sampling_percentage=100, 
    destination_s3_uri=s3_capture_upload_path # Path we defined earlier
)

# Deploy your trained estimator, NOT the generic 'model' object from the lab
predictor = xgb_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=endpoint_name,
    data_capture_config=data_capture_config, # Enable data capture
    serializer=CSVSerializer() # Add a serializer for sending data
)

print(f"\nEndpoint '{endpoint_name}' is now being created...")


# In[ ]:


#### 3.1 Execute predictions using the validation dataset.


# In[31]:


import numpy as np
from time import sleep

# Define a standard prediction threshold
prediction_cutoff = 0.5 
baseline_results_filename = "validation_with_predictions.csv"

# We will use the validation set (X_val, y_val) that's already in memory
# Let's limit it to 201 samples to match the lab's intent
limit = 201 
X_val_sample = X_val[:limit]
y_val_sample = y_val[:limit]

print(f"Generating predictions for {len(X_val_sample)} records...")

# Open a file to write the results
with open(baseline_results_filename, "w") as baseline_file:
    baseline_file.write("probability,prediction,label\n") # Write the header

    # Loop through our validation sample
    for i in range(len(X_val_sample)):
        
        # Get one row of features
        features_row = X_val_sample[i]
        
        # Send the feature row to the endpoint for a prediction.
        # The .predict() method handles serializing the numpy array to CSV.
        probability = float(predictor.predict(features_row))
        
        # Get the true label for this row
        label = y_val_sample[i]
        
        # Determine the predicted class based on our cutoff
        prediction = 1 if probability > prediction_cutoff else 0
        
        # Write the results to our file
        baseline_file.write(f"{probability},{prediction},{label}\n")
        
        print(".", end="", flush=True)
        sleep(0.5) # small sleep to simulate time between predictions

print("\nDone!")
print(f"Created baseline results file: {baseline_results_filename}")


# In[ ]:


#### 3.2 Examine the predictions from the model


# In[32]:


get_ipython().system('head validation_with_predictions.csv')


# In[33]:


baseline_prefix = prefix + "/baselining"
baseline_data_prefix = baseline_prefix + "/data"
baseline_results_prefix = baseline_prefix + "/results"

baseline_data_uri = f"s3://{bucket}/{baseline_data_prefix}"
baseline_results_uri = f"s3://{bucket}/{baseline_results_prefix}"
print(f"Baseline data uri: {baseline_data_uri}")
print(f"Baseline results uri: {baseline_results_uri}")


# In[34]:


from sagemaker.s3 import S3Uploader

# Upload the baseline results file you created earlier
baseline_dataset_uri = S3Uploader.upload(
    "validation_with_predictions.csv", 
    baseline_data_uri
)

print(f"Baseline dataset uploaded to: {baseline_dataset_uri}")


# In[ ]:


##### 3.4 Create a baselining job with validation dataset predictions


# In[35]:


from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor import EndpointInput
from sagemaker.model_monitor.dataset_format import DatasetFormat


# In[36]:


# Create the model quality monitoring object
# I've changed the variable name to reflect our project
crime_model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
    sagemaker_session=session,
)


# In[37]:


from datetime import datetime, timezone

# Use a name that reflects your project
baseline_job_name = f"crime-prediction-baseline-job-{datetime.now(timezone.utc):%Y-%m-%d-%H%M}"

print(f"Baseline Job Name: {baseline_job_name}")


# In[38]:


# Execute the baseline suggestion job
# This job analyzes your baseline data and suggests constraints.

# Use the monitor object we created for our project
job = crime_model_quality_monitor.suggest_baseline(
    job_name=baseline_job_name,
    baseline_dataset=baseline_dataset_uri,
    # The container correctly auto-detects the header, so no 'header=True' is needed
    dataset_format=DatasetFormat.csv(), 
    output_s3_uri=baseline_results_uri,
    problem_type="BinaryClassification",
    # These attributes tell the monitor which columns in your CSV to use
    inference_attribute="prediction",
    probability_attribute="probability",
    ground_truth_attribute="label",
)

print("Starting baselining job. This will take several minutes...")
job.wait(logs=False)
print("Baselining job complete.")


# In[ ]:


#### 3.5 Explore the results of the baselining job


# In[39]:


# The 'job' object from the last step has information about the completed job
baseline_job_description = job.describe()
baseline_job_output_s3_uri = baseline_job_description['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']

print(f"Baseline results are in: {baseline_job_output_s3_uri}")

# Find the exact path to the constraints file
from sagemaker.s3 import S3Downloader
import json

constraints_s3_path = os.path.join(baseline_job_output_s3_uri, "constraints.json")

# Download and print the constraints file
constraints_json = json.loads(S3Downloader.read_file(constraints_s3_path))

print("\n--- Generated Constraints ---")
# The 'pretty print' library makes the JSON easier to read
import pprint
pprint.pprint(constraints_json)


# In[40]:


# Get a reference to the baseline job we just ran
baseline_job = crime_model_quality_monitor.latest_baselining_job

# We can now access details from this object, for example:
print(f"Name of the completed baseline job: {baseline_job.job_name}")


# In[ ]:


#### 3.5.1 View the metrics generated


# In[41]:


binary_metrics = baseline_job.baseline_statistics().body_dict["binary_classification_metrics"]
pd.json_normalize(binary_metrics).T


# In[42]:


pd.DataFrame(baseline_job.suggested_constraints().body_dict["binary_classification_constraints"]).T


# In[ ]:


###Section 4 - Setup continuous model monitoring to identify model quality drif


# In[ ]:


#### 4.1 Generate prediction data for Model Quality  Monitoring


# In[43]:


from threading import Thread
from time import sleep
import boto3

# This function sends the contents of a file to the endpoint, row by row.
def invoke_endpoint(ep_name, file_name):
    # We are re-using the test data file we created for the batch job
    with open(file_name, "r") as f:
        print(f"Starting to send data from {file_name} to endpoint {ep_name}...")
        for i, row in enumerate(f):
            payload = row.rstrip("\n")
            
            # The 'predictor' object we created during deployment is easier to use
            predictor.predict(
                payload, 
                initial_args={"InferenceId": str(i)} # Provide a unique ID for each prediction
            )
            
            # sleep for a short time to simulate a realistic traffic pattern
            sleep(1)

# This function will loop forever, repeatedly sending the test data.
def invoke_endpoint_forever():
    while True:
        try:
            # Use the data file we know exists
            invoke_endpoint(endpoint_name, "test_batch_data.csv")
        except Exception as e:
            # Print any errors and continue
            print(f"Error invoking endpoint: {e}")
            sleep(10) # Wait a bit before retrying if there's an error

# Start sending data in a separate thread.
# This allows the notebook to remain interactive while data is sent in the background.
print("Starting background thread to continuously invoke endpoint...")
thread = Thread(target=invoke_endpoint_forever)
thread.daemon = True # This ensures the thread will close when the notebook kernel is shut down
thread.start()

print("✅ Background traffic generation started.")


# In[ ]:


### 4.2 View captured data


# In[44]:


print("Waiting for captures to show up", end="")
for _ in range(120):
    capture_files = sorted(S3Downloader.list(f"{s3_capture_upload_path}/{endpoint_name}"))
    if capture_files:
        capture_file = S3Downloader.read_file(capture_files[-1]).split("\n")
        capture_record = json.loads(capture_file[0])
        if "inferenceId" in capture_record["eventMetadata"]:
            break
    print(".", end="", flush=True)
    sleep(1)
print()
print("Found Capture Files:")
print("\n ".join(capture_files[-3:]))


# In[45]:


print("\n".join(capture_file[-3:-1]))


# In[46]:


print(json.dumps(capture_record, indent=2))


# In[ ]:


#### 4.3 Generate synthetic ground truth


# In[48]:


import random


def ground_truth_with_id(inference_id):
    random.seed(inference_id)  # to get consistent results
    rand = random.random()
    return {
        "groundTruthData": {
            "data": "1" if rand < 0.7 else "0",  # randomly generate positive labels 70% of the time
            "encoding": "CSV",
        },
        "eventMetadata": {
            "eventId": str(inference_id),
        },
        "eventVersion": "0",
    }


def upload_ground_truth(records, upload_time):
    fake_records = [json.dumps(r) for r in records]
    data_to_upload = "\n".join(fake_records)
    target_s3_uri = f"{ground_truth_upload_path}/{upload_time:%Y/%m/%d/%H/%M%S}.jsonl"
    print(f"Uploading {len(fake_records)} records to", target_s3_uri)
    S3Uploader.upload_string_as_file_body(data_to_upload, target_s3_uri)


# In[49]:


NUM_GROUND_TRUTH_RECORDS = 334  # 334 are the number of rows in data we're sending for inference


def generate_fake_ground_truth_forever():
    j = 0
    while True:
        fake_records = [ground_truth_with_id(i) for i in range(NUM_GROUND_TRUTH_RECORDS)]
        upload_ground_truth(fake_records, datetime.utcnow())
        j = (j + 1) % 5
        sleep(60 * 60)  # do this once an hour


gt_thread = Thread(target=generate_fake_ground_truth_forever)
gt_thread.start()


# In[ ]:


### 4.4 Create a monitoring schedule


# In[50]:


from datetime import datetime, timezone

# Use a name that reflects your project
monitoring_schedule_name = f"crime-prediction-monitoring-schedule-{datetime.now(timezone.utc):%Y-%m-%d-%H%M}"

print(f"Monitoring Schedule Name: {monitoring_schedule_name}")


# In[51]:


# Create an enpointInput
endpointInput = EndpointInput(
    endpoint_name=predictor.endpoint_name,
    probability_attribute="0",
    probability_threshold_attribute=0.5,
    destination="/opt/ml/processing/input_data",
)


# In[52]:


from sagemaker.model_monitor import CronExpressionGenerator

# Create the monitoring schedule to execute every hour.
# We are using the corrected monitor and schedule name variables.
crime_model_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name=monitoring_schedule_name,
    endpoint_input=endpointInput,
    output_s3_uri=s3_report_path, # Use the report path we defined
    constraints=baseline_job.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly(),
    enable_cloudwatch_metrics=True,
    
    # These next two arguments are for Model Quality monitoring
    problem_type="BinaryClassification",
    ground_truth_input=ground_truth_upload_path, 
)

print(f"Successfully created monitoring schedule: {monitoring_schedule_name}")


# In[53]:


# Use the monitor object for your project to describe the schedule
crime_model_quality_monitor.describe_schedule()


# In[ ]:


### 4.5 Examine monitoring schedule executions


# In[54]:


# Use the monitor object for your project
executions = crime_model_quality_monitor.list_executions()

if not executions:
    print("No executions found yet, as expected.")
    print("The first hourly job will be scheduled to run at the top of the next hour.")
else:
    print("Found executions:")
    print(executions)


# In[55]:


from time import sleep

# Use the monitor object for your project
print("Waiting for the first execution to start (this will be at the top of the hour, ~7:00 PM PDT)...", end="")
while True:
    execution = crime_model_quality_monitor.describe_schedule().get(
        "LastMonitoringExecutionSummary"
    )
    if execution:
        break
    print(".", end="", flush=True)
    sleep(10) # Check every 10 seconds
    
print()
print("✅ Execution found!")


# In[60]:


from time import sleep

# The 'execution' variable should already exist from a previous cell.
# Let's get the status from it.
status = execution["MonitoringExecutionStatus"]

# This loop will run as long as the job is Pending or InProgress
while status in ["Pending", "InProgress"]:
    print(f"Current execution status: '{status}'. Waiting for execution to finish...")
    
    # Wait for the underlying processing job to complete.
    latest_execution.wait(logs=False)
    
    # Describe the job again to get the latest details
    latest_job = latest_execution.describe()
    
    print()
    print(f"Processing Job Name: {latest_job.get('ProcessingJobName')}")
    print(f"--> Job Status: {latest_job.get('ProcessingJobStatus')}")
    print(f"--> Exit Message: {latest_job.get('ExitMessage')}")
    print(f"--> Failure Reason: {latest_job.get('FailureReason')}")
    print("-" * 50)
    
    # Wait a bit before checking the schedule status again
    sleep(30)
    
    # Use the corrected monitor object name here
    latest_execution = crime_model_quality_monitor.list_executions()[-1]
    execution = crime_model_quality_monitor.describe_schedule()["LastMonitoringExecutionSummary"]
    status = execution["MonitoringExecutionStatus"]

print(f"\nExecution finished with status: '{status}'")

if status != "Completed":
    print("\n====INVESTIGATE====")
    print(execution)
else:
    print("\n✅ Monitoring job completed successfully!")


# In[ ]:


### 4.5 View violations generated by monitoring schedule


# In[62]:


from sagemaker.s3 import S3Downloader
import json
import pprint

# --- Use the lab's method (with the corrected monitor name) ---
# Get the latest execution from our monitor
latest_execution = crime_model_quality_monitor.list_executions()[-1] 

# Get the S3 URI for the reports from the execution's description
report_uri = latest_execution.describe()["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
print("Found Report Uri:", report_uri)


# --- Now, find and print the violations.json file ---
print("\nSearching for 'violations.json' in the report directory...")

violation_files = S3Downloader.list(report_uri)
# Find the specific violations file in the list of all report files
violations_s3_path = [file for file in violation_files if file.endswith("violations.json")][0]

# Download and pretty-print the violations file
violations_json = json.loads(S3Downloader.read_file(violations_s3_path))

print("\n--- VIOLATIONS REPORT ---")
pprint.pprint(violations_json)


# In[ ]:


## Section 5 - Analyze model quality CloudWatch metrics 


# In[ ]:


## 5.1 List the CW metrics generated.


# In[64]:


import boto3

# Create CloudWatch client
cw_client = boto3.Session().client("cloudwatch")

# This is the standard namespace where SageMaker publishes monitoring metrics
namespace = "aws/sagemaker/Endpoints/model-metrics"

# Define the dimensions to filter metrics for your specific endpoint and schedule
cw_dimensions = [
    {"Name": "Endpoint", "Value": endpoint_name},
    # Use the schedule name from your project
    {"Name": "MonitoringSchedule", "Value": monitoring_schedule_name}, 
]


# In[65]:


# List metrics through the pagination interface
paginator = cw_client.get_paginator("list_metrics")

for response in paginator.paginate(Dimensions=cw_dimensions, Namespace=namespace):
    model_quality_metrics = response["Metrics"]
    for metric in model_quality_metrics:
        print(metric["MetricName"])


# In[ ]:


##  5.2 Create a CloudWatch Alarm


# In[66]:


# A descriptive name for the alarm
alarm_name = "Crime-Model-Quality-F2-Score-Alarm"
alarm_desc = (
    "Triggers an alarm when the F2 score for the crime model drifts below the threshold"
)

# Setting a purposefully low threshold to see the alarm trigger quickly.
# Our model's baseline F2 was ~0.67, so this will trigger.
f2_drift_threshold = 0.68

metric_name = "f2"
namespace = "aws/sagemaker/Endpoints/model-metrics"

cw_client.put_metric_alarm(
    AlarmName=alarm_name,
    AlarmDescription=alarm_desc,
    ActionsEnabled=False, # Set to False for the demo to avoid needing to set up an SNS topic.
    MetricName=metric_name,
    Namespace=namespace,
    Statistic="Average",
    Dimensions=[
        {"Name": "Endpoint", "Value": endpoint_name},
        # Use the schedule name from your project
        {"Name": "MonitoringSchedule", "Value": monitoring_schedule_name},
    ],
    Period=3600, # Corresponds to the hourly schedule
    EvaluationPeriods=1,
    DatapointsToAlarm=1,
    Threshold=f2_drift_threshold,
    ComparisonOperator="LessThanThreshold",
    # Treat missing data as a breach to be safe
    TreatMissingData="breaching", 
)


# In[ ]:


## 5.3 Validation


# In[ ]:


## Part 1: Setting Up a Data Quality Monitor
## The process will be very similar to what you just did: we will create a baseline, and then schedule a monitor.


# In[ ]:


# Step 1: Prepare and Upload the Baseline Dataset


# In[67]:


# The original 'X' DataFrame from before one-hot encoding should still be in memory 
# and contains the column names we need.
# Let's create a new DataFrame from our X_train data with the correct headers.
train_features_df = pd.DataFrame(X_train, columns=X.columns)

# Define a local path for the new baseline file
local_data_quality_baseline_path = 'data_quality_baseline.csv'

# Save the DataFrame to a CSV, this time WITH the header
train_features_df.to_csv(local_data_quality_baseline_path, index=False, header=True)

print(f"Created data quality baseline file: {local_data_quality_baseline_path}")
print(f"Shape of baseline data: {train_features_df.shape}")


# Define a new S3 prefix for this monitor's baseline data
data_quality_baseline_prefix = f"{prefix}/data-quality/baseline-data"
data_quality_baseline_s3_uri = f"s3://{bucket}/{data_quality_baseline_prefix}"

# Upload the baseline file to S3
data_quality_baseline_dataset_uri = S3Uploader.upload(
    local_data_quality_baseline_path, 
    data_quality_baseline_s3_uri
)

print(f"Data quality baseline dataset uploaded to: {data_quality_baseline_dataset_uri}")


# In[ ]:


# Part 1, Step 2: Create and Run the Data Quality Baselining Job


# In[69]:


from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from datetime import datetime, timezone

# === FIX and RE-RUN: Data Quality Baselining ===

# 1. Re-save the baseline file WITHOUT the header
train_features_df = pd.DataFrame(X_train, columns=X.columns)
local_data_quality_baseline_path = 'data_quality_baseline.csv'

# Note the change: header=False
train_features_df.to_csv(local_data_quality_baseline_path, index=False, header=False)

print(f"Re-saved data quality baseline file with header=False.")


# 2. Re-upload the headerless file to the same S3 location
data_quality_baseline_prefix = f"{prefix}/data-quality/baseline-data"
data_quality_baseline_s3_uri = f"s3://{bucket}/{data_quality_baseline_prefix}"
data_quality_baseline_dataset_uri = S3Uploader.upload(
    local_data_quality_baseline_path, 
    data_quality_baseline_s3_uri
)
print(f"Re-uploaded headerless baseline file to: {data_quality_baseline_dataset_uri}")


# 3. Re-run the baselining job with header=False
data_quality_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
    sagemaker_session=session,
)
data_quality_baseline_job_name = f"crime-data-quality-baseline-job-{datetime.now(timezone.utc):%Y-%m-%d-%H%M}"
data_quality_results_s3_uri = f"s3://{bucket}/{prefix}/data-quality/results"

print(f"\nStarting Data Quality Baselining job (v2): {data_quality_baseline_job_name}")
data_quality_job = data_quality_monitor.suggest_baseline(
    job_name=data_quality_baseline_job_name,
    baseline_dataset=data_quality_baseline_dataset_uri,
    # Note the change: header=False
    dataset_format=DatasetFormat.csv(header=False), 
    output_s3_uri=data_quality_results_s3_uri,
    wait=True,
    logs=True,
)

print("\nData Quality Baselining job complete.")


# In[ ]:


## Part 1, Step 3: Schedule the Data Quality Monitor


# In[72]:


from sagemaker.model_monitor import EndpointInput
from datetime import datetime, timezone

# === FIX and RE-RUN: Schedule the Data Quality Monitor (v2) ===

# 1. Create a NEW EndpointInput specifically for Data Quality.
data_quality_endpoint_input = EndpointInput(
    endpoint_name=endpoint_name,
    destination="/opt/ml/processing/input_data"
)

# 2. Now create the schedule using the correct input object and variable names.
print("Re-attempting to schedule the Data Quality Monitor...")

data_quality_baseline_job = data_quality_monitor.latest_baselining_job
data_quality_schedule_name = f"crime-data-quality-monitoring-schedule-{datetime.now(timezone.utc):%Y-%m-%d-%H%M}"

# Define the S3 path for the reports
data_quality_report_s3_path = f"s3://{bucket}/{prefix}/data-quality/reports"

# Create the hourly monitoring schedule for Data Quality
data_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name=data_quality_schedule_name,
    endpoint_input=data_quality_endpoint_input,
    output_s3_uri=data_quality_report_s3_path, # <-- This variable name now matches
    constraints=data_quality_baseline_job.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.hourly(),
    enable_cloudwatch_metrics=True,
)

print(f"\n✅ Successfully created Data Quality monitoring schedule: {data_quality_schedule_name}")


# In[ ]:


## Step 1: Define the Bias Analysis Configuration


# In[77]:


import json

# === Part 2: Setting Up a Model Bias Monitor (Manual & Final Approach) ===

print("Manually creating the bias analysis configuration file...")

# This Python dictionary defines the exact configuration needed by the
# SageMaker Clarify container. This is the most reliable method.
# It tells the job which columns to use from your 'validation_with_predictions.csv' file.
bias_config_dict = {
    "label": "label",
    "predicted_label": "prediction",
    "facet": [
        {
            "name_or_index": "Vict Sex_M",
            "value_or_threshold": [1]
        }
    ],
    "label_values_or_threshold": [1],
}

# Define a local filename for our manual configuration
bias_config_filename = "model_bias_config_manual.json"

# Save this dictionary to the JSON file
with open(bias_config_filename, "w") as f:
    json.dump(bias_config_dict, f)

print(f"\n✅ Bias analysis configuration manually created and saved to '{bias_config_filename}'.")


# In[78]:


from sagemaker.s3 import S3Uploader

# === Upload the Bias Config File to S3 ===

# Define a new S3 prefix for this monitor's configuration
model_bias_config_prefix = f"{prefix}/model-bias/config"

# Upload the local JSON file to S3
model_bias_config_uri = S3Uploader.upload(
    "model_bias_config_manual.json", # The local file we just created
    f"s3://{bucket}/{model_bias_config_prefix}"
)

print(f"✅ Bias analysis configuration uploaded to: {model_bias_config_uri}")


# In[ ]:


## Step 3: Create and Run the Model Bias Baselining Job


# In[81]:


# The 'clarify_processor' object should exist from the previous cell
help(clarify_processor.run_bias)


# In[83]:


import pandas as pd
from sagemaker.s3 import S3Uploader
from sagemaker.clarify import SageMakerClarifyProcessor, DataConfig, BiasConfig
from datetime import datetime, timezone

# === Part 2: Model Bias Monitor - Definitive Final Approach ===
# My sincere apologies again for the errors. This block corrects the root cause.

# Step 1: Create the correct, complete baseline dataset.
# It MUST contain the features, the predictions, and the true labels together.
print("Step 1: Creating the complete baseline dataset...")

# We need the original feature names.
feature_names = list(X.columns)

# We will use the test set, as that is what we have predictions for.
features_df = pd.DataFrame(X_test, columns=feature_names)
predictions_df = pd.DataFrame(y_pred_xgb, columns=["prediction"])
labels_df = pd.DataFrame(y_test, columns=["label"])

# Combine everything into a single DataFrame.
bias_baseline_df = pd.concat([features_df, predictions_df, labels_df], axis=1)
bias_baseline_local_path = "bias_baseline_complete.csv"

# Save this complete dataset to a new CSV file WITH the header.
bias_baseline_df.to_csv(bias_baseline_local_path, index=False, header=True)
print(f"Created complete baseline file '{bias_baseline_local_path}' with shape {bias_baseline_df.shape}")

# --------------------------------------------------------------------------

# Step 2: Upload this new, correct baseline file to S3.
print("\nStep 2: Uploading the complete baseline file to S3...")
bias_baseline_s3_uri = S3Uploader.upload(
    bias_baseline_local_path,
    f"s3://{bucket}/{prefix}/model-bias/baseline-data"
)
print(f"Uploaded to: {bias_baseline_s3_uri}")

# --------------------------------------------------------------------------

# Step 3: Create the necessary configuration objects.
print("\nStep 3: Creating configuration objects...")
bias_config = BiasConfig(
    label_values_or_threshold=[1],  # Favorable outcome is 1
    facet_name="Vict Sex_M",
    facet_values_or_threshold=[1], # Group of interest is 'Male' (where the column is 1)
)

model_bias_results_s3_uri = f"s3://{bucket}/{prefix}/model-bias/results"
data_config = DataConfig(
    s3_data_input_path=bias_baseline_s3_uri,
    s3_output_path=model_bias_results_s3_uri,
    headers=list(bias_baseline_df.columns),
    label='label',
    predicted_label='prediction',
    dataset_type="text/csv",
)
print("BiasConfig and DataConfig objects created.")

# --------------------------------------------------------------------------

# Step 4: Run the SageMaker Clarify job.
print("\nStep 4: Setting up and running the SageMaker Clarify job...")
clarify_processor = SageMakerClarifyProcessor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=session,
)

clarify_job_name = f"crime-clarify-bias-job-{datetime.now(timezone.utc):%Y-%m-%d-%H%M}"

clarify_processor.run_bias(
    data_config=data_config,
    bias_config=bias_config,
    model_config=None,
    post_training_methods="all",
    pre_training_methods=[], # Not running pre-training bias
    job_name=clarify_job_name,
    wait=True,
    logs=True,
)

print("\n✅ SageMaker Clarify bias analysis job complete.")


# In[85]:


from sagemaker.s3 import S3Downloader
import os

# === Download the Bias Report (Final Corrected Version) ===
# My apologies, this version corrects the list index error.

print("Finding the bias report S3 path...")

try:
    # Get the list of outputs from the latest job
    job_outputs = clarify_processor.latest_job.outputs

    # The analysis result is the first (and only) output in the list
    bias_report_output_path = job_outputs[0].s3_uri

    # The main report file is named 'report.pdf'
    bias_report_pdf_s3_path = f"{bias_report_output_path}/report.pdf"

    # Define the local filename to save the report as
    local_bias_report_path = "model_bias_report.pdf"

    print(f"Downloading bias report from: {bias_report_pdf_s3_path}")

    # Download the report from S3
    S3Downloader.download(
        s3_uri=bias_report_pdf_s3_path,
        local_path="." # Download to the current directory
    )

    # Rename the file for clarity, checking if it exists first
    if os.path.exists("report.pdf"):
        os.rename("report.pdf", local_bias_report_path)
        print(f"\n✅ Bias report downloaded successfully!")
        print(f"--> Please open the file '{local_bias_report_path}' from the file browser on the left to view the results.")
    else:
        print("\nERROR: 'report.pdf' was not found after download attempt.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please check the job status and S3 output path in the AWS console.")


# In[86]:


# The 'clarify_processor' object and its 'latest_job' should be in memory.
# Let's get the list of outputs again.
job_outputs = clarify_processor.latest_job.outputs

# The object causing the error is the first item in that list.
output_object = job_outputs[0]

# Now, let's print everything we can about this object to find the right property.
print("--- Object Summary ---")
print(output_object)

print("\n\n--- All Available Properties and Methods ---")
print(dir(output_object))


# In[87]:


from sagemaker.s3 import S3Downloader
import os

# === Download the Bias Report (Definitive Final Version) ===
# Using the '.destination' property that we discovered from your environment.

print("Finding the bias report S3 path using the correct property...")

try:
    # Get the list of outputs from the latest job
    job_outputs = clarify_processor.latest_job.outputs

    # Access the first output object and use the .destination attribute
    bias_report_output_path = job_outputs[0].destination

    # The main report file is named 'report.pdf'
    bias_report_pdf_s3_path = f"{bias_report_output_path}/report.pdf"
    local_bias_report_path = "model_bias_report.pdf"

    print(f"Downloading bias report from: {bias_report_pdf_s3_path}")

    # Download the report from S3
    S3Downloader.download(
        s3_uri=bias_report_pdf_s3_path,
        local_path="." # Download to the current directory
    )

    # Rename the file for clarity, checking if it exists first
    if os.path.exists("report.pdf"):
        os.rename("report.pdf", local_bias_report_path)
        print(f"\n✅ Bias report downloaded successfully!")
        print(f"--> Please open the file '{local_bias_report_path}' from the file browser on the left to view the results.")
    else:
        print("\nERROR: 'report.pdf' was not found after download attempt.")

except Exception as e:
    print(f"\nAn error occurred: {e}")


# In[ ]:




