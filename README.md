# Crimewise Analytics: LAPD Resource Optimization using MLOps
Machine Learning Operation (AAI-540-02)

![Conceptual Diagram for Public Safety Infrastructure](https://publications.aecom.com/social-infrastructure/static/3344e1ded97f7807c31b7c3449c4fac2/fd45d/3de1cb069b30a7880613e4d6646a6588.webp "Public Safety Infrastructure Overview")

## Project Overview
The Los Angeles Police Department (LAPD) is grappling with significant operational hurdles, including projected civilian staff layoffs in 2025 due to budgetary constraints from recent wildfires and an ongoing transition to the National Incident-Based Reporting System (NIBRS). These challenges underscore a critical need for optimized resource allocation to maintain public safety effectively.

This project, under the umbrella of Crimewise Analytics, a public safety consulting firm specializing in data-driven solutions, addresses this need head-on. We're developing a robust Machine Learning Operations (MLOps) solution to empower the LAPD with proactive, data-driven insights. Our primary focus is on predicting the broad category of crime (Person, Property, or Society) most likely to occur at specific times and locations. This capability will enable the LAPD to strategically deploy its limited personnel, anticipate resource demands, and ultimately improve efficiency, response times, and public safety outcomes.

## Business Problem
The core business challenge facing the LAPD is how to sustain and enhance law enforcement effectiveness with a significantly reduced civilian workforce, including key crime analysts. The concurrent NIBRS upgrade, while promising more granular data, also presents immediate integration and analytical challenges. Crimewise Analytics aims to bridge this analytical gap by providing an automated, intelligent system that ensures officers are deployed where and when they are most needed, maximizing the impact of limited human resources.

## Technical Details

### Model Objectives
The primary objective of this project is to build a multi-class classification system that predicts the crime_against category (Person, Property, or Society) for a given incident. This system will leverage temporal, locational, and offense-related features derived from NIBRS data, providing the LAPD with a high-level, actionable understanding of potential criminal activity for optimized resource allocation.

### Machine Learning Problem Type
This project explicitly tackles a multi-class classification problem. The model will classify a crime incident into one of three predefined, broad categories: Crime Against Person, Crime Against Property, or Crime Against Society.

### Data Source
The project primarily utilizes publicly available crime data from the City of Los Angeles Open Data Portal:

Crime_Data_from_2020_to_Present.csv: This serves as the foundational dataset, containing crime incident reports from 2020 to the present. It includes data collected under both the legacy Summary Reporting System (SRS) and the newer National Incident-Based Reporting System (NIBRS) after March 2024. This transition within the dataset itself presents a realistic MLOps challenge to address data consistency. Two supplementary datasets will be used for prediction: (1) LAPD_NIBRS_Offenses_Dataset.csv; and (2) 	Arrest_Data_from_2010_to_2019.csv

## Data Preparation and Feature Engineering

The raw data undergoes a comprehensive cleaning and feature engineering process to prepare it for machine learning, focusing on features relevant to predicting the crime_against category:

* Missing Value Handling: Columns with very high percentages of missing data (e.g., Weapon Used Cd, Crm Cd 4) are dropped. Other missing values (e.g., Vict Age, various categorical features) are handled through imputation (median for numerical, mode for categorical).
* Location Data Cleaning: Rows with invalid geographical coordinates (0°, 0°) are removed.
* Temporal Feature Engineering: DATE OCC and TIME OCC are parsed, and new features such as hour_of_day, day_of_week, month, year, and weekend are extracted to capture temporal patterns.
* Categorical Feature Encoding: Categorical features like AREA_NAME, Premis_Desc, Vict Sex, Vict Descent, and Mocodes (after handling missing values) are transformed using One-Hot Encoding to make them suitable for machine learning algorithms.
* Target Variable Creation & Encoding: The original Crm Cd Desc is used to derive the new, higher-level crime_against target variable (Person, Property, Society). This derived target is then transformed using Label Encoding for multi-class classification.

## Data Exploration

Exploratory Data Analysis (EDA) was performed to gain insights into the dataset's characteristics and inform feature selection:

* Initial insights into DataFrame shape, data types, and missing value distributions.
* Analysis of the distribution of the newly created crime_against categories.
* Visualization of Vict Age, LAT, LON distributions and their relation to crime types.
* Analysis of top crime types (Crm Cd Desc), police areas (AREA_NAME), and crime statuses (Status).
* Geographical scatter plots to visualize crime locations and potential hotspots.
* Box plots to understand victim age distribution by police area and original crime type.

## Hypothesized Main Features

Based on the EDA, domain understanding, and the NIBRS structure, we hypothesize that the most influential features for predicting the crime_against category will include:

* Temporal Features: hour_of_day, day_of_week, month, weekend. These are crucial as different crime types exhibit distinct temporal patterns.
* Spatial Features: AREA_NAME, Rpt_Dist_No, LAT, LON (or derived spatial bins/clusters). Crime types often cluster geographically.
* Premise Features: Premis_Desc. The type of location (e.g., residential, commercial, public space) is highly indicative of the nature of the crime.
* Victim Demographics: Vict Age, Vict Sex, Vict Descent. These can correlate with crimes against persons versus property/society, though their use requires careful ethical consideration.

## Model Selection

For the multi-class classification problem of predicting crime_against category, we plan to explore and compare:

* Benchmark Model: A simple heuristic model that predicts the most frequent crime_against category observed in the training data. This will provide a fundamental baseline for evaluating the performance of more complex models.

* Advanced ML Model: A RandomForestClassifier. This ensemble model is chosen for its robustness, ability to handle high-dimensional and non-linear data, and strong performance in many classification tasks. Its interpretability through feature importances is also a  benefit.

## Model Evaluation
Model performance will be rigorously evaluated using standard classification metrics, focusing on the crime_against prediction:

* Accuracy: The overall proportion of correctly predicted crime_against categories.
* Precision (for each class): The proportion of positive identifications that were actually correct for each crime_against category.
* Recall (Sensitivity) (for each class): The proportion of actual positives that were correctly identified for each crime_against category.
* F1-Score (for each class): The harmonic mean of precision and recall, particularly useful for potentially imbalanced classes within the crime_against categories.
* Classification Report: Provides a detailed breakdown of precision, recall, and F1-score for each class, allowing for granular performance analysis.
* Comparison against Benchmark: The advanced model's performance will be directly compared against the benchmark model on the held-out test set to quantify its added value and efficacy.

## Project Structure and Workflow

This project adheres to MLOps principles by defining clear stages for data collection, preparation, modeling, and evaluation. The development workflow is iterative and designed for eventual deployment in a cloud environment.

1.  Data Collection: Raw Crime_Data_from_2020_to_Present.csv downloaded from the City of Los Angeles Open Data Portal. (Future: Stored in an S3 Data Lake for scalable access).
2.  Data Preparation & Feature Engineering: Performed in a Python environment (Google Colab for initial development and prototyping, transitioning to AWS SageMaker notebooks for cloud-based development). This includes cleaning, imputation, creation of temporal/spatial features, and the crucial step of deriving and encoding the crime_against target variable. (Future: Engineered features will be stored in a SageMaker Feature Store for centralized management and access).
3.  Data Splitting: The processed and feature-engineered data is strategically split into Training (~40%), Validation (~10%), Test (~10%), and a dedicated "Production Data" (~40%) set. A time-based split is rigorously applied to ensure the model's evaluation reflects its ability to predict future, unseen crime events accurately.
4.  Model Training:
    * Benchmark Model: A simple Most-Frequent-Class Classifier, trained on the training data.
    * Advanced ML Model: A RandomForestClassifier, trained on the comprehensive training feature set
5. Model Evaluation: Performance metrics are calculated on the validation and test sets, and compared against the benchmark to assess the model's predictive power and generalizability.

## Installation and Setup
To replicate and run this project, you will need:
* Python 3.x
* Google Colab (highly recommended for ease of setup and experimentation) or a local Python environment.
* Required Python libraries (installable via pip):
* * pandas
  * numpy
  * scikit-learn
  * collections

### You can install these dependencies by running:
* pip install pandas numpy scikit-learn
* !wget https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD -O Crime_Data_from_2020_to_Present.csv
  

#### To download the primary dataset directly into your Colab environment, execute the following in a Colab cell:
* !wget https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD -O Crime_Data_from_2020_to_Present.csv

  ## Goals vs. Non-Goals

  ### Goals:
* Develop a functional multi-class classification model capable of predicting the crime_against category with demonstrable accuracy against a benchmark.
* Implement a robust data preprocessing and feature engineering pipeline tailored for crime incident data, including the derivation of the crime_against target.
* Establish a structured and realistic data splitting strategy (training, validation, test, production-simulated) using a time-based approach.
* Quantify the performance of the advanced model using standard classification metrics and clearly compare its efficacy to a simple baseline.
* Provide a foundational MLOps structure (demonstrated through pipeline stages) that can be extended for future cloud deployment and continuous integration/delivery.

  ### Non-Goals:
* Achieving state-of-the-art predictive accuracy that would be required for immediate, live, mission-critical operational deployment without further extensive tuning, optimization, and real-time data integration.
* Developing a fully integrated, real-time production system with live API endpoints, sophisticated dashboards, or automated retraining loops within this project's initial scope.
* Conducting exhaustive hyperparameter tuning for the advanced model; the focus is on a reasonable baseline performance to prove concept.
* Integrating external data sources beyond the provided LAPD crime dataset (e.g., socio-economic indicators, weather data) for this phase of the project.
* Performing deep dives into advanced Natural Language Processing (NLP) techniques on free-text fields like Mocodes beyond basic handling.
* Conducting a comprehensive ethical auditing of the model's potential biases and fairness implications; however, potential data biases will be acknowledged and discussed at a high level.
