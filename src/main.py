import pandas as pd
import geopandas as gp

from shapely.geometry import Point
from sklearn import linear_model, metrics

from railway_graph import RailwayGraph
from transforms import RailwayGraphOneHotWithLength


if __name__ == '__main__':
    # Загружаем информацию о станциях
    stations_geo_df: gp.GeoDataFrame = gp.read_file('../data/stations_v2/stationsv2.shp')
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
    railway_graph = RailwayGraph(
        '../data/all_routes_v2/all_routes_v2.shp',
        full_stations_geo_df,
    )

    # Загружаем тестовые данные
    train_df = pd.read_csv('../data/dataset_external.csv')
    test_df = pd.read_csv('../data/dataset_internal.csv')

    # Создаём предобработку
    data_transforms = RailwayGraphOneHotWithLength(railway_graph, undirectional=True, no_found="skip")
    # Предобрабатываем данные
    train_y = train_df.values
    train_X = data_transforms.fit_transform(train_df)
    test_X = data_transforms.transform(test_df)

    # Обучаем модель
    model = linear_model.Ridge()
    model.fit(train_X, train_y)

    # Считаем качество обобщение информации
    train_pred = model.predict(train_X)
    print('MAE:', metrics.mean_absolute_error(y, train_pred))
    print('MAPE:', metrics.mean_absolute_percentage_error(y, train_pred))

    # Получаем предсказание
    test_pred = model.predict(test_X)

    submit_df = test_df.copy()
    submit_df['value_predict'] = test_pred

    submit_df.to_csv('../submition_boost_700_oh_with_len.csv', index=False)
