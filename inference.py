import argparse

import joblib

from src.feature_preprocessing import feature_preprocessing


def main(
    command_args=None,
    features=None,
    # model_path="models/baseline_model.joblib",
    model_path="models/production_model.joblib",
):
    # Если аргументы переданы из командной строки, парсим их
    print(command_args)
    if command_args is not None:
        age = command_args.age
        systolic_bp = command_args.systolic_bp
        diastolic_bp = command_args.diastolic_bp
        temperature = command_args.temperature
        heart_rate = command_args.heart_rate
        glucose_level = command_args.glucose_level
        model_path = command_args.model_path
    else:
        age, systolic_bp, diastolic_bp, temperature, heart_rate, glucose_level = (
            features
        )

    features = feature_preprocessing(
        age, systolic_bp, diastolic_bp, temperature, heart_rate, glucose_level
    )

    model = joblib.load(model_path)

    probabilities = model.predict_proba(features)

    print(probabilities)

    return probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Инференс скрипт для предсказания рисков для беременных"
    )
    parser.add_argument("age", type=float, help="Значение age")
    parser.add_argument("systolic_bp", type=float, help="Значение systolic_bp")
    parser.add_argument("diastolic_bp", type=float, help="Значение diastolic_bp")
    parser.add_argument("temperature", type=float, help="Значение temperature")
    parser.add_argument("heart_rate", type=float, help="Значение heart_rate")
    parser.add_argument("glucose_level", type=float, help="Значение glucose_level")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/baseline_model.joblib",
        help="Путь к файлу с моделью",
    )
    args = parser.parse_args()

    main(command_args=args)
