import numpy as np
import pandas as pd


def get_pressure_group(*args):
    low_bp = args[2]
    high_bp = args[1]
    if high_bp < 110 or low_bp < 70:
        return "hypotension"
    elif (high_bp >= 110 and high_bp < 120) and (low_bp >= 70 and low_bp < 80):
        return "optimal"
    elif (high_bp >= 110 and high_bp < 130) or (low_bp >= 70 and low_bp < 85):
        return "normal"
    elif (high_bp >= 130 and high_bp < 139) or (low_bp >= 85 and low_bp < 89):
        return "high_normal"
    elif (high_bp >= 140 and high_bp < 159) or (low_bp >= 90 and low_bp < 99):
        return "hypertension_stage_1"
    elif (high_bp >= 160 and high_bp < 179) or (low_bp >= 100 and low_bp < 109):
        return "hypertension_stage_2"
    elif high_bp >= 180 or low_bp >= 110:
        return "hypertension_stage_3"


def get_age_group(*args):
    age = args[0]
    if age < 20:
        return "younger age"
    elif age > 35:
        return "older age"
    else:
        return "medium age"


def feature_preprocessing(*args):
    feature0 = get_age_group(*args)
    feature1 = get_pressure_group(*args)
    feature_vector = np.array([feature0, feature1] + list(args), dtype=object)
    feature_df = pd.DataFrame(
        feature_vector.reshape(1, -1),
        columns=[
            "age_group",
            "pressure_group",
            "age",
            "systolic_bp",
            "diastolic_bp",
            "glucose_level",
            "body_temperature",
            "heart_rate",
        ],
    )
    return feature_df
