# Перечень и описание исследований

## Исследование № 1. Эксперименты по повышению качества бейзлайн модели.

В рамках исследования проведен анализ важности признаков бейзлайн модели:
   > - Самый важный признак для класса "Высокий риск" - группа артериального давления 'normal', что весьма странно. Вероятно, имеется слишком много записей с другими признаками, реально влияющими на риск, но с нормальным давлением. Кроме того, среди признаков отсутствуют некоторые группы, которые не были представлены в данных после разбиения на выборки из-за малого размера датасета. 
   >   - Целесообразно пересмотреть способ разделения на группы по артериальному давлению, например на три группы: "нормальное", "повышенное", "пониженное".
   >   - Целесообразно рассмотреть вопрос влияния регуляризации (пропробуем её убрать).
   > - Зачастую  признак, равноценный для двух классов, например, температура тела или диастолическое давление, играет малую роль для третьего класса. Вероятно, это связано с малой представленностью признака среди записей различных классов. Фактически, это сказывается малый объем данных. Целесообразно попробовать применить техники оверсемплинга.
   > - На основании ранее проведенного EDA и взаимной корреляции количественных признаков артериального давления предлагается провести эксперименты с исключение одного или обоих признаков, а также использованием их совокупности, например, произведения.
   > - Предлагается исключить признак ЧСС, как обладающий наименьшей важностью для всех классов. 

На основании анализа важности признаков принято решение и проведены следующие эксперименты:
1. Исключена регуляризация, осуществлен подбор гиперпараметров.
2. Реализован оверсемплинг с использованием SMOTENC.
3. Изменен признак групп по давлению.
4. Удален один количественный признак давления ("верхнее").
5. Удален один количественный признак давления ("нижнее").
6. Удалены оба количественных признака давления.
7. Использована их совокупность - произведение.
8. Удален признак ЧСС. 
9. Использованы оригинальные признаки без признаков групп давления и возраста.

### Результат исследования № 1
В результате экспериментов не удалось улучшить качество моделей. Принято решение использовать иные модели.

## Исследование № 2. Эксперименты по применению иных моделей.

В данном исследовании (эксперименте) применены алгоритмы, основанные на решающих деревьях, а именно:
1. Классический DesicionTreeClassifier().
2. Ансамблевый RandomForestClassifier().

Осуществлено обучение с подбором гиперпараметров.

Результат качества лучшей модели получен: F1-Weighted = 0.749 с использованием RandomForestClassifier().

Модель, показавшая лучший результат, чем бейзлайн модель после подбора гиперпараметров была сохранена.

## Исследование № 3. Эксперименты по применению иных моделей.

В данном исследовании (эксперименте) применен алгоритм градиентного бустинга, реализованный в библиотеке CatBoost.

Осуществлено обучение с подбором гиперпараметров.

Результат качества лучшей модели получен: F1-Weighted = 0.759.

Модель, показавшая лучший результат, чем бейзлайн модель после подбора гиперпараметров была сохранена и выбрана в качестве лучшей финальной модели.

## Исследование № 4. Эксперименты с лучшей моделью с использованием оверсемплинга.

В данном исследовании (эксперименте) применена техника оверсемплинга для лучшей модели, построенной с применением алгоритма градиентного бустинга.

Результат качества лучшей модели получен: F1-Weighted = 0.742.

Эксперимент не продемонстрировал лучших результатов. Обученная модель не подлежит сохранению.

## Финальное исследование. Оценка качества финальной модели на тестовой выборке.

В данном исследовании (эксперименте) мы оценено качество работы лучшей модели на тестовой выборке, изучены ошибки модели, а также обучена модель на полном датасете для внедрения в приложение.