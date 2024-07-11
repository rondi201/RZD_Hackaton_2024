# Решение трека "Прогнозирование сроков доставки товара"

Решение представлено для [задачи хакатона](https://cogmodel.mipt.ru/iprofitrack2), проводимого в рамках форума "Я-Профессионал" в МФТИ в 2024 году.

Более подробно о решении можно узнать из [презентации](https://docs.google.com/presentation/d/1AOrpcpAtrpYdylI5Wn6odMm3sWCqJ3cY7WE0Rp_lssk/edit#slide=id.g2eb913685b7_2_222).

## Установка

### 1. Создание среды

Для создания среды запуска предлагается использовать библиотеку [virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

```shell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Установка зависимостей

Установку зависимостей можно произвести с помощью [requirements.txt](requirements.txt).

```shell
pip install -r requirements.txt
```

### 3. Загрузка данных

Для работы необходимо загрузить данные `all_routes_v2.zip` и `stations_v2.zip`, после чего разархивировать их в папки `.data/all_routes_v2` и `.data/stations_v2`. 

Кроме того, также необходимо получить обучающие `dataset_external.csv` и предсказываемые `dataset_internal.csv` данные и поместить их в папку `data/`. 

Получить данные можно у организаторов хакатона.


## Использование

Для предсказания времени транспорта в пути достаточно запустить главную программу следующим образом:

```shell
python -m src.main
```

Более подробно о настройке параметров запуска можно познакомиться через `python -m src.main -h`