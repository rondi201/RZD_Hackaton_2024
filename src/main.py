import logging
import pandas as pd
import geopandas as gp
import argparse

from shapely.geometry import Point
from sklearn import metrics, ensemble
from pathlib import Path

from railway_graph import RailwayGraph
from transforms import RailwayGraphOneHotWithLength

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--routes', help='path to routes .shp file', type=str,
                        default=Path(ROOT, 'data', 'all_routes_v2', 'all_routes_v2.shp'))
    parser.add_argument('-s', '--stations', help='path to stations .shp file', type=str,
                        default=Path(ROOT, 'data', 'stations_v2', 'stationsv2.shp'))
    parser.add_argument('-t', '--train', help='path to train (external) .csv data file', type=str,
                        default=Path(ROOT, 'data', 'dataset_external.csv'))
    parser.add_argument('-p', '--predict', help='path to predict (internal) .csv data file', type=str,
                        default=Path(ROOT, 'data', 'dataset_internal.csv'))
    parser.add_argument('-o', '--output', help='prediction result data save path', type=str,
                        default=Path(ROOT, 'runs', 'submission.csv'))

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)

    # Загружаем информацию о станциях
    logging.info('Load stations geo info')
    stations_geo_df: gp.GeoDataFrame = gp.read_file(opt['stations'])
    # Добавляем данные о неизвестных станциях
    extra_stations_geo_df = gp.GeoDataFrame(
        {
            'name': [
                'Александров-2',
                'Брянск-Льговский',
                'Люблино-Сортировочное',
                'Орехово-Зуево',
                'Стенькино-I',
                'Тула 2'
            ],
            'geometry': [
                Point(38.6913863, 56.3721772),
                Point(34.4094874, 53.2135764),
                Point(37.7300731, 55.6715844),
                Point(38.9756838, 55.7953125),
                Point(39.7378679, 54.5343002),
                Point(37.5567752, 54.1693498),
            ]
        },
        crs='EPSG:4326'
    )
    # Объединяем станции
    full_stations_geo_df = pd.concat(
        [
            stations_geo_df,
            extra_stations_geo_df
        ],
        axis=0
    )

    # Загружаем граф маршрутов
    logging.info('Load routes geo info and build routes graph')
    railway_graph = RailwayGraph(
        opt['routes'],
        full_stations_geo_df,
    )

    # Загружаем тестовые данные
    logging.info('Load train and predict data')
    train_df = pd.read_csv(opt['train'])
    test_df = pd.read_csv(opt['predict'])

    # Создаём предобработку
    data_transforms = RailwayGraphOneHotWithLength(railway_graph, undirectional=True, no_found="skip")
    # Предобрабатываем данные
    train_y = train_df['value']
    logging.info('Transform train data')
    train_X = data_transforms.fit_transform(train_df)
    logging.info('Transform predict data')
    test_X = data_transforms.transform(test_df)

    # Обучаем модель
    logging.info('Train model')
    model = ensemble.GradientBoostingRegressor(n_estimators=700)
    model.fit(train_X, train_y)

    # Считаем качество обобщение информации
    train_pred = model.predict(train_X)
    print('Full train MAE:', metrics.mean_absolute_error(train_y, train_pred))
    print('Full train MAPE:', metrics.mean_absolute_percentage_error(train_y, train_pred))

    # Получаем предсказание
    logging.info('Predict by model')
    test_pred = model.predict(test_X)

    # Записываем результат в таблицу
    logging.info('Save predict result')
    submit_df = test_df.copy()
    submit_df['value_predict'] = test_pred

    # Сохраняем результат в файл
    Path(opt['output']).parent.mkdir(exist_ok=True, parents=True)
    submit_df.to_csv(opt['output'], index=False)
