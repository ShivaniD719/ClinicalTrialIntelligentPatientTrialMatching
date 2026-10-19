AI Clinical Trial Intelligence Platform

## Problem Statement:
For my capstone project, I am exploring how AI and machine learning could help match patients with relevant clinical trials. Finding a suitable trial can be difficult because trial information includes medical terms, different names for the same condition, and detailed eligibility requirements.

For this project, I collected clinical trial data, explored its structure, cleaned and analyzed key fields, and trained an introductory text-classification baseline. I have not yet built or validated a patient-specific trial eligibility matching model.

### Dataset

I collected study records from the public ClinicalTrials.gov API. I searched for lung cancer and retrieved 10,000 study records using paginated API requests.

API query: https://clinicaltrials.gov/api/v2/studies

The query above describes the intended search and maximum sample size. In my notebook, I fetched the records in smaller batches to avoid downloading one very large response.

I have curated dataset to 18 fields so I could retain information that may help with patient–trial matching without uploading the full raw API response.

|Curated field       |Original API module / field                                      |Why I kept it                                                                               |
|--------------------|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------|
|`nct_id`            |`identificationModule.nctId`                                     |Unique study identifier and duplicate checks                                                |
|`title`             |`identificationModule.briefTitle`                                |Trial topic and text modeling                                                               |
|`status`            |`statusModule.overallStatus`                                     |Recruitment availability and EDA                                                            |
|`study_type`        |`designModule.studyType`                                         |Compare observational and interventional studies                                            |
|`enrollment`        |`designModule.enrollmentInfo.count`                              |Enrollment analysis and outliers                                                            |
|`conditions`        |`conditionsModule.conditions`                                    |Disease terminology and relevance proxy                                                     |
|`sex`               |`eligibilityModule.sex`                                          |Basic eligibility information                                                               |
|`minimum_age`       |`eligibilityModule.minimumAge`                                   |Basic eligibility information                                                               |
|`maximum_age`       |`eligibilityModule.maximumAge`                                   |Basic eligibility information                                                               |
|`eligibility_text`  |`eligibilityModule.eligibilityCriteria`                          |**Full inclusion and exclusion criteria**, important for future matching                    |
|`healthy_volunteers`|`eligibilityModule.healthyVolunteers`                            |Additional eligibility context                                                              |
|`study_population`  |`eligibilityModule.studyPopulation`                              |Population descriptions                                                                     |
|`phase`             |`designModule.phases`                                            |Trial development stage                                                                     |
|`enrollment_type`   |`designModule.enrollmentInfo.type`                               |Actual versus estimated counts                                                              |
|`brief_summary`     |`descriptionModule.briefSummary`                                 |More context for text analysis                                                              |
|`intervention_names`|`armsInterventionsModule.interventions[].name`                   |Treatment information                                                                       |
|`locations`         |`contactsLocationsModule.locations[]` (city, state, country)     |Potential geographic matching                                                               |
|`last_update_posted`|`statusModule.studyFirstPostDateStruct.date` in the export script|**Currently contains first-posted date, despite its name**; not used in the current analysis|

I chose these fields based on the project scope. The curated dataset does not contain every source field. I kept the original raw JSON as a separate backup for future extraction.

How I collected and saved the data

I originally retrieved records directly from the API into Python memory using requests and pagination, then extracted fields from the nested JSON. This is how I obtained the data; the submitted notebook does not make a live API request on every run.

The original raw JSON was about 300 MB, which was too large for a regular GitHub commit. I therefore used a one-time export to select the 18 fields above and save them as data/clinical_trials_curated.json.gz. This retains the full eligibility-criteria text while reducing the dataset size. I used the fixed file for the submitted analysis to keep the data consistent for re-execution if needed.

### Analysis

1. Loading and checking the data

I initially used API response to extract fields. For the final notebook, I load the fixed compressed JSON file with gzip and json, then create a pandas DataFrame. This avoids network-dependent execution.

My initial checks showed:

|Check                                   |Result|
|----------------------------------------|-----:|
|Total records                           |10,000|
|Initial EDA columns                     |10    |
|Curated source fields for final notebook|18    |
|Unique NCT IDs                          |10,000|
|Duplicate NCT IDs                       |0     |

2. Checking missing values

I used pandas to count missing values in each column. Some of the main results were:

|Field      |Missing values|
|-----------|-------------:|
|Maximum age|7,094         |
|Minimum age|395           |
|Enrollment |166           |
|Sex        |4             |

I kept missing ages and enrollment values as missing rather than filling them with arbitrary numbers. For example, an unspecified maximum age should not automatically be treated as an average age.

I also converted age strings such as 18 Years into numeric values in years for future analysis. I kept the original age columns as well.

3. Creating additional features

I created a few features to make the data easier to explore:

● Number of conditions listed for each trial

● Combined condition text for keyword searches

● Number of words in the eligibility criteria

● Numeric minimum and maximum ages

● A flag showing whether a condition field matches selected lung cancer terms

These features may also be useful when I start developing a matching approach.

4. Exploring lung cancer terminology
As the data comes from a search result, I checked if every returned record is specifically about lung cancer.

My first keyword search identified 7,697 of 10,000 records (76.97%) as having selected lung cancer terms in their condition fields.

After looking at examples that did not match, I noticed terms such as Lung Carcinoma, Cancer of the Lung, and NSCLC. I expanded the keyword list to include more terms and abbreviations.

|Keyword matching approach|Records matched|
|-------------------------|--------------:|
|Initial terms            |7,697          |
|Expanded terms           |8,196          |
|Additional records found |499            |

The expanded list matched 81.96% of the records, leaving 1,804 without a match in the condition field. This does not prove that the remaining records are unrelated to lung cancer; some may use other terms or mention lung cancer elsewhere in the study information.

This is one reason I want to explore NLP for patient–trial matching later. The counts above measure keyword coverage, not clinical matching accuracy.

5. Exploring enrollment and outliers

I used Seaborn histograms and boxplots to examine enrollment. The initial plots were difficult to read because a small number of studies had very large participant counts. I then used a log(1 + enrollment) transformation for visualization and compared enrollment by study type.

|Enrollment measure              |Result              |
|--------------------------------|-------------------:|
|Records with reported enrollment|9,834               |
|Median enrollment               |70                  |
|Mean enrollment                 |Approximately 1,816|
|Maximum enrollment              |5,940,299           |
|Records with zero enrollment    |328                 |

The large difference between the mean and median suggests that enrollment is strongly right-skewed.

I also compared study types:

|Study type    |Records with enrollment|Median|Mean  |Maximum  |
|--------------|----------------------:|-----:|-----:|--------:|
|Interventional|7,691                  |60    |301   |200,000  |
|Observational |2,143                  |172   |72,656|5,940,299|

Observational studies had much larger extreme values. I have not removed these outliers, because large registry or observational studies may legitimately include many participants. The original enrollment values are preserved; the log transformation was used for plotting.

6. Understanding zero and missing enrollment

I checked the study statuses for records with zero enrollment. All 328 records with zero enrollment had a status of WITHDRAWN. I kept these records for EDA because they help describe the full dataset, but I would not recommend withdrawn studies as currently recruiting trials.

I then investigated the 166 records with missing enrollment:

|Study type     |Missing enrollment|
|---------------|-----------------:|
|Interventional |116               |
|Expanded access|40                |
|Observational  |10                |

The missing records had these statuses:

|Status                   |Count|
|-------------------------|----:|
|Completed                |87   |
|Unknown                  |34   |
|No longer available      |19   |
|Approved for marketing   |12   |
|Available                |7    |
|Terminated               |5    |
|Temporarily not available|2    |

I decided not to replace missing enrollment with zero because missing information is different from a reported value of zero. Expanded access records may also have different reporting patterns from conventional trials.

### Data-Cleaning Decisions

|Finding                    |Decision so far                                                           |
|---------------------------|--------------------------------------------------------------------------|
|Duplicate NCT IDs          |None found; no removal needed                                             |
|Missing age values         |Preserve as missing until eligibility rules are defined                   |
|Missing enrollment         |Preserve as missing; do not replace with zero                             |
|Zero enrollment            |Keep in EDA; exclude withdrawn trials from active matching recommendations|
|Very large enrollment      |Keep for now and investigate rather than automatically delete             |
|Different lung cancer terms|Use expanded keywords for exploration; improve further later              |

### Modeling 
Baseline Classification Model and Evaluation

As an introductory machine learning exercise, I prepared a TF-IDF + Logistic Regression classifier to predict whether a trial will receive the expanded lung-cancer keyword flag (lung_cancer_match_v2). The target label comes from the trial’s condition field. The model uses the study title and eligibility text as its inputs; it does not use the condition field as an input, to avoid simply copying the rule that created the label.

I chose this approach because the study information contains a lot of text and I wanted to try a straightforward text-classification method before exploring more advanced NLP. TF-IDF turns text into numeric features, and Logistic Regression learns patterns associated with each class. I also included a majority-class DummyClassifier so I can see whether the model improves on a simple baseline.

Data Modeling

1. Combine the trial title and eligibility criteria into a text field, filling missing text with an empty string.

2. Exclude records where both text fields are empty.

3. Split the data into 80% training / 20% testing, using stratification and random_state=42.

4. Fit the TF-IDF vectorizer only on training data by using a scikit-learn pipeline.

5. Train a Logistic Regression classifier, and compare it with a majority-class dummy model.

6. Evaluate both models on the held-out test set using accuracy, precision, recall, and F1-score. Plot a confusion matrix for Logistic Regression.

Main metric: F1-score for the positive keyword-match class. F1 balances precision (how many predicted matches were labeled matches) and recall (how many labeled matches were found). Accuracy alone can be misleading because most trials already belong to the positive class. I will also inspect class-specific results and the confusion matrix.

Evaluation results

I evaluated both models on a held-out test set of 2,000 records (20% of the dataset). The positive class is keyword match.

|Model                       |Accuracy|Precision (match)|Recall (match)|F1 (match)|Macro F1|
|----------------------------|-------:|----------------:|-------------:|---------:|-------:|
|Majority-class dummy        |0.820   |0.820            |1.000         |0.901     |0.45    |
|TF-IDF + Logistic Regression|0.801   |0.921            |0.828         |0.872     |0.71    |

The dummy classifier predicted keyword match for every trial. It scored slightly higher on accuracy and positive-class F1, but did not identify any non-matching records. Logistic Regression achieved a higher macro F1 and better performance on the minority class (no keyword match: precision 0.46, recall 0.68, F1 0.55). I learned why checking both classes is important when labels are imbalanced.

The Logistic Regression confusion matrix (actual labels as rows, predictions as columns) was:

|Actual / Predicted|No match|Match|
|------------------|-------:|----:|
|No match          |245     |116  |
|Match             |282     |1,357|

This means the classifier correctly predicted 1,602 of 2,000 proxy labels. It missed 282 keyword-matching records and incorrectly predicted a match for 116 non-matching records. I would not describe this model as better overall without specifying the metric: it improves minority-class recognition and macro F1, but not accuracy or positive-class F1.




Notebook

Jupyter notebook: https://github.com/ShivaniD719/ClinicalTrialIntelligentPatientTrialMatching/blob/main/IntelligentClinicalTrialPatientMatching.ipynb


Tools Used

Python, pandas, NumPy, gzip/json, Seaborn, Matplotlib, scikit-learn, Jupyter Notebook, and the ClinicalTrials.gov API (requests was used for the original one-time data collection).
