"""
Group 4 - Student Dropout Prediction
Synthetic data generation script (exactly as provided in the task brief).
Do NOT modify the generation logic - this file exists so the dataset
is 100% reproducible from source, as required by the task rules.
"""
import numpy as np
import pandas as pd
import os


def generate_group4_dataset(save_path="data/raw/group4_dataset.csv"):
    np.random.seed(88)
    n = 5000
    attendance = np.random.uniform(30, 100, n)
    study_hours = np.random.uniform(0, 15, n)
    assignments = np.random.uniform(20, 100, n)
    lms_activity = np.random.lognormal(5, 2, n)

    academic_index = (
        0.04 * attendance
        + 0.10 * study_hours
        + 0.03 * assignments
        + 0.0005 * lms_activity
    )

    dropout = (academic_index < np.median(academic_index)).astype(int)

    df = pd.DataFrame({
        "attendance": attendance,
        "study_hours": study_hours,
        "assignment_score": assignments,
        "lms_activity": lms_activity,
        "academic_index": academic_index,
        "dropout": dropout
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_group4_dataset()
    print(df.head())
    print(df.shape)
    print(df["dropout"].value_counts())
