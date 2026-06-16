import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

n = 500

study_hours = np.random.randint(1, 11, n)
attendance = np.random.randint(50, 101, n)
assignments = np.random.randint(1, 6, n)
previous_score = np.random.randint(30, 101, n)
age = np.random.randint(16, 23, n)
gender = np.random.randint(0, 2, n)
internet_access = np.random.randint(0, 2, n)
family_income = np.random.randint(1, 6, n)
sleep_hours = np.random.randint(4, 10, n)
extracurricular = np.random.randint(0, 2, n)
exam_prep_hours = np.random.randint(1, 21, n)
parent_education = np.random.randint(1, 6, n)

performance_score = (
    study_hours * 4
    + attendance * 0.3
    + assignments * 5
    + previous_score * 0.4
    + internet_access * 5
    + family_income * 2
    + sleep_hours * 2
    + extracurricular * 2
    + exam_prep_hours * 2
    + parent_education * 2
)

pass_status = (performance_score > 85).astype(int)

df = pd.DataFrame({
    "StudyHours": study_hours,
    "Attendance": attendance,
    "Assignments": assignments,
    "PreviousScore": previous_score,
    "Age": age,
    "Gender": gender,
    "InternetAccess": internet_access,
    "FamilyIncome": family_income,
    "SleepHours": sleep_hours,
    "Extracurricular": extracurricular,
    "ExamPrepHours": exam_prep_hours,
    "ParentEducation": parent_education,
    "Pass": pass_status
})

X = df.drop("Pass", axis=1)
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("      STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 60)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nEnter Student Details")
print("-" * 30)

study_hours = float(input("Study Hours per Day: "))
attendance = float(input("Attendance Percentage: "))
assignments = int(input("Assignments Completed (1-5): "))
previous_score = float(input("Previous Exam Score: "))
age = int(input("Age: "))
gender = int(input("Gender (0=Female, 1=Male): "))
internet_access = int(input("Internet Access (0=No, 1=Yes): "))
family_income = int(input("Family Income Level (1-5): "))
sleep_hours = float(input("Sleep Hours per Day: "))
extracurricular = int(input("Extracurricular Activities (0=No, 1=Yes): "))
exam_prep_hours = float(input("Exam Preparation Hours: "))
parent_education = int(input("Parent Education Level (1-5): "))

new_student = pd.DataFrame({
    "StudyHours": [study_hours],
    "Attendance": [attendance],
    "Assignments": [assignments],
    "PreviousScore": [previous_score],
    "Age": [age],
    "Gender": [gender],
    "InternetAccess": [internet_access],
    "FamilyIncome": [family_income],
    "SleepHours": [sleep_hours],
    "Extracurricular": [extracurricular],
    "ExamPrepHours": [exam_prep_hours],
    "ParentEducation": [parent_education]
})

prediction = model.predict(new_student)[0]
probability = model.predict_proba(new_student)[0]

print("\n" + "=" * 60)
print("                 PREDICTION RESULT")
print("=" * 60)

print(f"Study Hours          : {study_hours}")
print(f"Attendance           : {attendance}%")
print(f"Assignments          : {assignments}")
print(f"Previous Score       : {previous_score}")
print(f"Age                  : {age}")
print(f"Sleep Hours          : {sleep_hours}")
print(f"Exam Prep Hours      : {exam_prep_hours}")

print("\nFinal Result")

if prediction == 1:
    print("Status               : PASS")
    print(f"Confidence           : {probability[1] * 100:.2f}%")
    print("Performance          : Good Academic Standing")
else:
    print("Status               : FAIL")
    print(f"Confidence           : {probability[0] * 100:.2f}%")
    print("Performance          : Needs Improvement")

print("=" * 60)

print("\nPerformance Insights")
print("-" * 30)

if study_hours < 4:
    print("• Increase daily study hours.")

if attendance < 75:
    print("• Improve attendance percentage.")

if previous_score < 50:
    print("• Focus on strengthening core subjects.")

if sleep_hours < 6:
    print("• Get adequate sleep for better learning.")

if exam_prep_hours < 5:
    print("• Spend more time preparing for exams.")

if (
    study_hours >= 4 and
    attendance >= 75 and
    previous_score >= 50 and
    sleep_hours >= 6 and
    exam_prep_hours >= 5
):
    print("• Excellent overall performance indicators.")

print("\nThank you for using the Student Performance Prediction System.")