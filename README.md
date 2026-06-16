# Student Performance Prediction System

## Overview

The Student Performance Prediction System is a Machine Learning project developed using Python and Scikit-learn. The system predicts whether a student is likely to pass or fail based on various academic, personal, and environmental factors.

This project demonstrates the complete machine learning workflow, including data generation, preprocessing, model training, evaluation, and prediction.

---

## Features

* Predicts student performance (Pass/Fail)
* Uses multiple student-related factors for prediction
* Generates a dataset of 500 student records
* Trains a Decision Tree Classifier
* Displays model accuracy and classification report
* Accepts student details for real-time prediction
* Provides personalized performance insights and recommendations

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

## Dataset Features

The model uses the following features:

| Feature         | Description                                 |
| --------------- | ------------------------------------------- |
| StudyHours      | Daily study hours                           |
| Attendance      | Attendance percentage                       |
| Assignments     | Number of assignments completed             |
| PreviousScore   | Previous exam score                         |
| Age             | Student age                                 |
| Gender          | Student gender                              |
| InternetAccess  | Availability of internet access             |
| FamilyIncome    | Family income level                         |
| SleepHours      | Average sleep hours per day                 |
| Extracurricular | Participation in extracurricular activities |
| ExamPrepHours   | Hours spent preparing for exams             |
| ParentEducation | Parent education level                      |

Target Variable:

* Pass = 1
* Fail = 0

## Machine Learning Model

Algorithm Used:

* Decision Tree Classifier

Model Parameters:

* Criterion: Entropy
* Max Depth: 6
* Minimum Samples Split: 5
* Random State: 42

## Project Workflow

1. Generate student dataset
2. Create feature and target variables
3. Split data into training and testing sets
4. Train Decision Tree model
5. Evaluate model performance
6. Accept student inputs
7. Predict student result
8. Display confidence score and insights

## Sample Output

      STUDENT PERFORMANCE PREDICTION SYSTEM

Model Accuracy: 94.00%

Prediction Result

Status               : PASS
Confidence           : 96.43%
Performance          : Good Academic Standing

## Future Improvements

* Use a real-world student dataset
* Add data visualization dashboards
* Implement Random Forest and XGBoost models
* Build a web application using Flask
* Deploy the project on the cloud
* Add student performance analytics
* 
## Learning Outcomes

This project helps in understanding:

* Data preprocessing
* Feature engineering
* Classification algorithms
* Model evaluation
* Machine learning workflow
* Python-based data science development
