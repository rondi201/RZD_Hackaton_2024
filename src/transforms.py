import numpy as np
import pandas as pd
import sklearn

from tqdm.auto import tqdm
from typing import Literal

from src.railway_graph import RailwayGraph


class RailwayGraphOneHotWithLength(sklearn.base.TransformerMixin):
    def __init__(
            self,
            graph: RailwayGraph,
            verbose: bool = True,
            undirectional: bool = False,
            no_found: Literal["skip", "error"] = "error"
    ):
        self._graph = graph
        self._vebose = verbose
        self._edge_2_id_map = {}
        self._fit = False
        self._undirectional = undirectional
        self._no_found = no_found

    def fit(self, data: pd.DataFrame, y=None):
        self._fit = True
        return self

    def transform(self, data: pd.DataFrame, y=None):
        if not self._fit and not self._edge_2_id_map:
            raise ValueError('run fit before transform')

        num_samples = len(data['route_start'])
        len_route = np.zeros(num_samples, dtype=np.uint32)
        if self._fit:
            oh_route = np.zeros((num_samples, len(self._graph.edges)), dtype=bool)
        else:
            oh_route = np.zeros((num_samples, len(self._edge_2_id_map)), dtype=bool)

        # Получим допустимые для кодирования станции
        available_stations = set(self._graph.stations)
        # Пройдёмся по всем маршрутам
        data_iter = zip(data['route_start'], data['route_end'])
        if self._vebose:
            data_iter = tqdm(data_iter, total=num_samples)
        for i, (src, dst) in enumerate(data_iter):
            # Проверим на то, что станции доступны для обработки
            for name in (src, dst):
                if name not in available_stations:
                    raise ValueError(f"'{name}' not found in available stations")
            # Найдём кратчайший маршрут между станциями
            route = self._graph.find_route(src, dst)
            edges_info = self._graph.get_route_geodata(route)
            for u, v, k in edges_info.index:
                if self._undirectional and u > v:
                    u, v = v, u
                edge_idx = (u, v, k)
                if edge_idx not in self._edge_2_id_map:
                    if self._fit:
                        self._edge_2_id_map[edge_idx] = len(self._edge_2_id_map)
                    elif self._no_found == "error":
                        raise ValueError(f"Edge with index '{edge_idx}' not found in fit dataset")
                    else:
                        continue

                oh_route[i][self._edge_2_id_map[edge_idx]] = True
            len_route[i] = edges_info.length.sum()

        if self._fit:
            oh_route = oh_route[:, :len(self._edge_2_id_map)]

        len_data = pd.DataFrame(
            len_route[:, None],
            columns=['length'],
        )
        oh_data = pd.DataFrame(
            oh_route,
            columns=(f'route_{i}' for i in range(oh_route.shape[1]))
        )
        data = pd.concat([len_data, oh_data], axis=1)

        self._fit = False

        return data
