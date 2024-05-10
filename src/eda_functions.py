import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phik  # noqa: F401
import scipy.stats as stats
import seaborn as sns
from matplotlib.axes import Axes


def plot_pie(data: np.ndarray, axes: Axes, legend: str) -> None:
    """
    Строит круговую диаграмму для данных.

    Parameters
    ----------
    data : numpy.ndarray
        Входные данные для построения круговой диаграммы.
    axes : matplotlib.axes.Axes
        Объект осей для построения.
    legend : str
        Подпись для круговой диаграммы.

    Returns
    -------
    None
    """
    data.value_counts().plot(
        kind="pie",
        autopct="%1.0f%%",
        ylabel="",
        title="Соотношение исследуемого признака.\nСтолбец " + legend,
        ax=axes,
    )


def analysis_cat_cols(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Строит графики анализа категориальных признаков.
    Если количество категорией меньше 5 - строит круговую диаграмму.
    Если количество категорий больше или равно 5 - строит горизонтальную
    столбчатую диаграмму.

    Parameters
    ----------
    df : pandas.DataFrame
        Дата-сет для которого проводится анализ.
    columns : list of str
        Список анализируемых столбцов дата-сета df.

    Returns
    -------
    None
        Результат выводится на экран.
    """

    # Список для столбцов для круговой диаграммы
    col_for_pie = []
    # Список для столбцов для горизонтальной столбчатой диаграммы
    col_for_barh = []

    def pie_plot(x: int) -> bool:
        """
        Определение, подходит ли круговая диаграмма.
        Будем строить круговую диаграмму только если
        количество уникальных значений меньше 5
        """
        return True if x < 5 else False

    # Перебираем все столбцы для анализа
    for col in columns:
        # Если количество уникальных значений меньше 5,
        # добавляем столбец в список для круговой диаграммы.
        # Иначе добавляем столбец в список для столбчатой диаграммы
        if pie_plot(df[col].nunique()):
            col_for_pie.append(col)
        else:
            col_for_barh.append(col)

    length = len(col_for_pie)
    rows = int(-(-length // 4))

    # Если есть только один столбец для круговой диаграммы, строим ее
    if length == 1:
        legend = "[" + col_for_pie[0] + "]"
        plot_pie(df[col_for_pie[0]], None, legend)
    # Если есть несколько столбцов для круговой диаграммы
    # Строим фигуру с несколькими диаграммами
    elif rows > 1:
        # Создаем фигуру максимум с 4 диаграммами в строке
        cols = 4 if ((length % 4 != 1) or (length % 3 == 1)) else 3
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))

        # Строим круговую диаграмму для каждого столбца
        for i in range(0, len(col_for_pie)):
            legend = "[" + col_for_pie[i] + "]"
            plot_pie(df[col_for_pie[i]], axes[i // cols, i % cols], legend)

        # Удаляем лишние подграфики
        for k in range(i + 1, cols * rows):
            fig.delaxes(axes[k // cols][k % cols])

        fig.tight_layout()

    plt.show()

    # Строим столбчатые диаграммы для столбцов,
    # не подходящих для круговой диаграммы
    for i in range(0, len(col_for_barh)):
        legend = "[" + col_for_barh[i] + "]"
        df[col_for_barh[i]].value_counts(ascending=True).plot(
            kind="bar",
            title="Соотношение исследуемого признака.\nСтолбец " + legend,
            ylabel="",
        )
        plt.xlabel("Количество элементов")
        plt.ylabel("Категория")
        plt.show()

    return None


def describe_numcols(
    df,
    column,
    desc=("hist", "QR", "describe", "box", "eject"),
    hist_desc=(
        "Распределение\nисследуемого параметра",
        "Единицы изменения\nисследуемого параметра",
        "Частота повторений\nисследуемого параметра",
    ),
    box_desc=(
        "Диаграмма размаха\nисследуемого параметра",
        "Наименование\nисследуемого столбца",
        "Единицы изменения\nисследуемого параметра",
    ),
):
    """
    Функция, выводящая основные показатели для анализа количественных
    признаков

    Parameters
    ----------
    df : pandas.DataFrame
        дата-сет для которого проводится анализ
    column : str
        столбец, для которого проводится анализ
    desc : list, optional
        способы вывода результатов анализа,
        by default ['hist', 'QR', 'describe', 'box', 'eject']
    hist_desc : list, optional
        подписи для гистограммы,
        by default ['Распределение\nисследуемого параметра',
        'Единицы изменения\nисследуемого параметра',
        'Частота повторений\nисследуемого параметра']
    box_desc : list, optional
        подписи для диаграммы размаха,
        by default ['Диаграмма размаха\nисследуемого параметра',
        'Наименование\nисследуемого столбца',
        'Единицы изменения\nисследуемого параметра']

    Returns
    -------
    None
        результат выводится на экран

    """

    legend_my_desc = "Столбец" + " [" + column + "]"

    if "hist" in desc:
        bins = int(np.round(math.sqrt(len(df[column])), 0))
        df[column].plot(kind="hist", bins=bins, grid=True, legend=True)
        plt.title(hist_desc[0])
        plt.xlabel(hist_desc[1])
        plt.ylabel(hist_desc[2])
        plt.legend([legend_my_desc])
        plt.show()

    quartiles = df[column].quantile([0.25, 0.75])
    iqr = quartiles[0.75] - quartiles[0.25]
    min_diagram = max(quartiles[0.25] - 1.5 * iqr, df[column].min())
    max_diagram = min(quartiles[0.75] + 1.5 * iqr, df[column].max())

    if "describe" in desc:
        print()
        print("Ключевые характеристики распределения:")
        print(df[column].describe().apply("{0:.2f}".format))

    if "QR" in desc:
        print()
        print("Ключевые характеристики диаграммы размаха:")
        print("IQR\t\t\t{0:.2f}".format(iqr))
        print("Q1-1,5*IQR\t\t{0:.2f}".format(quartiles[0.25] - 1.5 * iqr))
        print("MIN диаграммы размаха\t{0:.2f}".format(min_diagram))
        print("Q3+1,5*IQR\t\t{0:.2f}".format(quartiles[0.75] + 1.5 * iqr))
        print("MAX диаграммы размаха\t{0:.2f}".format(max_diagram))
        print()

    if "box" in desc:
        df[column].plot(kind="box", grid=True, legend=False)
        plt.title(box_desc[0])
        plt.xlabel(box_desc[1])
        plt.ylabel(box_desc[2])
        plt.show()

    if "eject" in desc:
        print("Ключевые характеристики выбросов:")
        print(
            "Количество выбросов (вверх)\t{0:.0f}".format(
                len(df[df[column] > max_diagram])
            )
        )
        print(
            "Доля выбросов (вверх)\t\t{0:.2f}%".format(
                len(df[df[column] > max_diagram]) / len(df) * 100
            )
        )
        print(
            "Количество выбросов (вниз)\t{0:.0f}".format(
                len(df[df[column] < min_diagram])
            )
        )
        print(
            "Доля выбросов (вниз)\t\t{0:.2f}%".format(
                len(df[df[column] < min_diagram]) / len(df) * 100
            )
        )


def corr_matrix(df, cols, corr_type, target=None, interval_cols=None):
    """
    Функция, формирующая матрицу корреляции

    :param df: pandas.Series; распределение, для которого проводится анализ
    :param cols: list
    :param corr_type: str, 'phik' или 'spearman' или ''
    :param target: str
    :return: None; результат выводится на экран

    """

    new_cols = list(cols)
    if corr_type == "phik":
        new_cols.append(target)
        corr = df[new_cols].phik_matrix(interval_cols=interval_cols, njobs=-1)
    else:
        corr = df[new_cols].corr(method=corr_type)
    plt.figure(figsize=(30, 15))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", annot=True, fmt=".2f")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45, va="top")
    plt.title("Матрица корреляции")
    plt.show()
    return corr


def is_norm(data):
    """
    Функция, позволяющая оценить нормальность распределения количественной
    величины

    :param data: pandas.Series; распределение, для которого проводится анализ
    :return: None; результат выводится на экран

    """

    res = stats.normaltest(data)
    stats.probplot(data, dist="norm", plot=plt)
    plt.show()
    # Коэффициент статистической значимости установим 0.05
    if res.pvalue > 0.05:
        return (
            "p-value = {}. Имеются основания говорить "
            "о нормальности распределения".format(res.pvalue)
        )
    else:
        return (
            "p-value = {}. Имеются основания говорить "
            "о НЕнормальности распределения".format(res.pvalue)
        )
