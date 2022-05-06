Taya homework for RS School Machine Learning course.

This homework uses [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

## How repeat my results
This package allows you to train model for detecting the cover type of the forest.
1. Clone this repository to your machine.
2. Download [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save csv locally (default path is *data/train.csv* in repository's root).
[pic](https://disk.yandex.ru/i/OVLsGcZz82PNEw)
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine.
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install
```
```sh
poetry add pandas
```
```sh
poetry add black
```
```sh
poetry add flake8
```
```sh
poetry add mlflow
```
```sh
poetry add sklearn
```
```sh
poetry add scikit-learn
```
```sh
poetry add joblib
```
```sh
poetry add --dev mypy
```
Add package to dev dependencies for testing:
```sh
poetry add --dev pytest
```
5. List of Tasks:
5.1. I used the Forest train dataset.
5.2. I formated my homework as a Python package.
5.3. I publish code to Github with > 30 commits. **(12 points)**
5.4. I used poetry to manage my package and dependencies. **(6 points)** And split installed dependencies for development and not development. **(4 points)**
5.5. Add my data to gitignore. **(5 points)** Create EDA report but it costs 0 point. Script in Profiling.py, file name DataFrameProfile.html in the root directory. 
5.6. I wrote a script that trains a model and saves it to a file. I use click when create CLI. **(10 points)** And I registered script in pyproject.toml. **(2 points)**  Random seeds by default = 42.
Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
First I use the same Logistic Regression like in demo. 
```sh
poetry run train
```
[terminal](https://disk.yandex.ru/i/3FfS6YlUtlJL6A)
Run MLflow UI to see the result:
```sh
poetry run mlflow ui
```
[mlflow](https://disk.yandex.ru/i/_hwLJt1YEBPF_A)
5.7. I use K-fold cross-validation. And I create choice of 2 metrics. **(10 points)** 
[mlflow](https://disk.yandex.ru/i/bfA_JwR8ovNG5w)
5.8.
If you choose number 1 then train Logistic Regression (and by default), if you choose number 2 then train RandomForestClassifier. **(4 points)** I try at least three different sets of hyperparameters for each model. **(3 points)** If you choose number 1 in --use-feature-selection then apply SelectFromModel, if you choose number 2 then apply VarianceThreshold. If you choose another number or not use feature-selection flag then you does not apply any feature-selection. **(4 points)**
My experiments:
```sh
poetry run train --ml-model 1 --max-iter 1000 --logreg-c 10 --use-feature-selection 1
```
```sh
poetry run train --ml-model 1 --max-iter 2000 --logreg-c 100 --use-feature-selection 1
```
```sh
poetry run train --ml-model 1 --max-iter 700 --logreg-c 0.5 --use-feature-selection 1
```
```sh
poetry run train --ml-model 1 --max-iter 1000 --logreg-c 1 --use-feature-selection 1
```
```sh
poetry run train --ml-model 1 --max-iter 1000 --logreg-c 10 --use-feature-selection 2
```
```sh
poetry run train --ml-model 1 --max-iter 2000 --logreg-c 100 --use-feature-selection 2
```
```sh
poetry run train --ml-model 1 --max-iter 3000 --logreg-c 200 --use-feature-selection 2
```
```sh
poetry run train --ml-model 2 --n-estimators 10 --use-feature-selection 1
```
```sh
poetry run train --ml-model 2 --n-estimators 150 --use-feature-selection 1
```
```sh
poetry run train --ml-model 2 --n-estimators 200 --use-feature-selection 1
```
```sh
poetry run train --ml-model 2 --n-estimators 10 --use-feature-selection 2
```
```sh
poetry run train --ml-model 2 --n-estimators 150 --use-feature-selection 2
```
```sh
poetry run train --ml-model 2 --n-estimators 200 --use-feature-selection 2
```
Run MLflow UI to see the information about experiments:
```sh
poetry run mlflow ui
```
[mlflow](https://disk.yandex.ru/i/O_80_G4_TAYVbg)
5.9. I use nested cross-validation with GridSearchCV, but maybe my computer very weak, CPU over 100% and I cant't wait end of executing script. But you can see code of it and you have choose to use or not GridSearchCV. Default value --grid-search = False. **(10 points)**
```sh
poetry run train --ml-model 1 --use-feature-selection 2 --grid-search False
poetry run train --ml-model 1 --use-feature-selection 2 --grid-search True
poetry run train --ml-model 2 --use-feature-selection 2 --grid-search False
poetry run train --ml-model 2 --use-feature-selection 2 --grid-search True
```
5.10. I wrote this README. **(10 points)** + **(2 points)**

5.11. Only 1 test error and 1 test success without using fake/sample data and filesystem isolation, as in the demo. **(3 points)**
```sh
poetry run pytest
```
[terminal](https://disk.yandex.ru/i/KXHC_2dNt2v8Ng)
5.12. The code in this repository formatted with black and lint with flake8. **(2 points)**
```sh
poetry run black <namefile.py>
```
[terminal](https://disk.yandex.ru/i/BEDZN42bQ7ngCw)
[terminal](https://disk.yandex.ru/i/Yiz_g1z9vAGxmQ)
[terminal](https://disk.yandex.ru/i/sAyNSz8V2lMXiQ)
flake8 interfere black. Black create E501 line too long when I check flake8.
[terminal](https://disk.yandex.ru/i/uj0V3mtt5jDt7w)
[terminal](https://disk.yandex.ru/i/kNKIjIo2N3OZBQ)
So...
```sh
poetry run flake8 --ignore=E501 <namefile.py>
```
[terminal](https://disk.yandex.ru/i/EM7SUT6KtPZk5w)
[terminal](https://disk.yandex.ru/i/vr_RlY96OuxV-w)
5.13. I pass mypy typechecking. **(3 points)**
```sh
poetry run mypy <namefile.py>
```
[terminal](https://disk.yandex.ru/i/M1HFug595WNfYQ)
[terminal](https://disk.yandex.ru/i/M0myvGh4d5ZHDQ)
5.14. I combine steps of testing and linting into a single command using nox session. **(2 points)**
I not use install_with_constraints because of "Permission denied: 'C:\\Users\\T3BB0~1.TIT\\AppData\\Local\\Temp\\tmp1kty95iy'"
And I switch on show_error_codes = True, but it not help to get the error code. 
I use session.run("mypy", "--warn-unused-ignores", *args), but it not help to ignore the error. 
I use # type: ignore[unused-ignore] and # type: ignore[no-untyped-def, unused-ignore] but it not help to ignore the error. 
I put away strict mode but it not help. 
But if I use only poetry run mypy then all successful.
```sh
poetry run nox
```
[terminal](https://disk.yandex.ru/i/KruqP87YL8GpcA)
[terminal](https://disk.yandex.ru/i/YemI2UaOdtDZJA)
