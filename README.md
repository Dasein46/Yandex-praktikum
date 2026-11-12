# Data Science: учебные проекты и исследования

Портфолио учебных проектов, выполненных в рамках программы Data Science Яндекс Практикума. Темы: классификация и регрессия, временные ряды, NLP, статистический анализ и Computer Vision.

> **О воспроизводимости:** четыре исследования ниже получили обновлённые протоколы валидации и автоматизированные синтетические проверки. После изменений полные ноутбуки на исходных данных **не запускались**; старые метрики из отдельных README относятся к прежним версиям и не являются результатами нового кода. Подробности — в [METHODOLOGY.md](METHODOLOGY.md).

## Избранные ML-проекты

| Кейс | Бизнес-задача | Подход и проверка |
| --- | --- | --- |
| [Прогноз заказов такси](prediction-taxi-orders/) | Предсказать спрос на следующий час для планирования числа водителей | Лаги и скользящие признаки, LinearRegression / DecisionTree / XGBoost, TimeSeriesSplit, RMSE и baseline lag 24 |
| [Отток клиентов банка](customer-outflow/) | Выделить клиентов с риском ухода | Классификация, обработка дисбаланса, train/validation/test, F1 и ROC-AUC |
| [Токсичные комментарии](nlp-project/) | Помочь модерации выявлять токсичный текст | Очистка текста, TF-IDF внутри CV, классификаторы, F1 |
| [Восстановление золота](gold-recovery/) | Прогнозировать показатели восстановления металла на этапах производства | Регрессия, Pipeline, TimeSeriesSplit, взвешенная sMAPE |

## Другие проекты

| Направление | Проекты |
| --- | --- |
| Регрессия | [Стоимость автомобиля](price-of-cars/), [Выбор региона нефтедобычи](choosing-well-location/) |
| Классификация и CV | [Рекомендация тарифа](recommendation-of-tariffs/), [Определение возраста по фото](age-of-buyers/) |
| Аналитика и статистика | [Тарифы телеком-компании](tariff-for-telecom-company/), [Продажи игр](game-sales-research/), [Недвижимость](sale-of-apartments/), [Надёжность заёмщиков](reliability-of-borrowers/), [Музыка двух городов](yandex-music/) |
| Преобразование данных | [Защита персональных данных](protection-of-personal-data/) |

## Технологии

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, NLTK, TF-IDF, TensorFlow/Keras (в отдельных проектах).

## Методология и запуск

Для четырёх обновлённых ML-исследований документированы разделения данных, процедуры оценки, ограничения и тесты: **[методология и воспроизводимость](METHODOLOGY.md)**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python -m unittest discover -s tests -v
```

Для запуска ноутбуков нужны учебные датасеты, не включённые в репозиторий. Укажите `DATA_DIR` и откройте соответствующий ноутбук; список файлов и оговорки приведены в методологии.

Другие проекты автора: [Computer Vision](https://github.com/Dasein46/My_projects) · [Olist: SQL и Power BI](https://github.com/Dasein46/olist-analysis) · [Маркетплейс: DataLens](https://github.com/Dasein46/marketplace-inventory-analysis).
