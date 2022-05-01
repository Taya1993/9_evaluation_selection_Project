"""импорт библиотек"""
import pandas as pd
import pandas_profiling

"""загружаем датасет"""
df = pd.read_csv('data/train.csv')

"""создаём отчёт"""
profile = df.profile_report(title='Pandas Profiling Report', progress_bar=False)
"""смотрим отчёт в HTML формате"""
profile.to_file("DataFrameProfile.html")