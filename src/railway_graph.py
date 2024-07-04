import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gp
import osmnx as ox
import pandas as pd

from shapely.geometry import LineString
from pathlib import Path
from tqdm.auto import tqdm


def convert_gpd2graph(gdf, make_G_bidi=True, name='unamed'):
    """
    Converts shapefile to routable networkx graph.
    Based on https://gist.github.com/philippschw/a75f2436308c776d78057407eae76904

    Parameters
    ----------
    p : str, File path - allowed formats geojson and ESRI Shapefile and other formats Fiona can read and write
    make_G_bidi : bool, if True, assumes linestrings are bidirectional
    name : str, Optional name of graph

    Returns
    -------
    G : graph
    """
    gdf = gdf.copy()

    # shapefile needs to include minimal: geometry linestring and the length computed (e.g. in QGIS)
    if 'length' not in gdf.columns:
        raise Exception('Shapefile is invalid: length not in attributes:\n{}'.format(gdf.columns))

    if not gdf.geometry.map(lambda x: type(x) == LineString).all():
        s_invalid_geo = gdf.geometry[gdf.geometry.map(lambda x: type(x) == LineString)]
        raise Exception('Shapefile is invalid: geometry not all linestring \n{}'.format(s_invalid_geo))

    # Compute the start- and end-position based on linestring
    gdf['Start_pos'] = gdf.geometry.apply(lambda x: x.coords[0])
    gdf['End_pos'] = gdf.geometry.apply(lambda x: x.coords[-1])

    # Create Series of unique nodes and their associated position
    s_points = pd.concat([gdf.Start_pos, gdf.End_pos]).reset_index(drop=True)
    s_points = s_points.drop_duplicates()
    #     log('GeoDataFrame has {} elements (linestrings) and {} unique nodes'.format(len(gdf),len(s_points)))

    # Add index of start and end node of linestring to geopandas DataFrame
    df_points = pd.DataFrame(s_points, columns=['Start_pos'])
    df_points['FNODE_'] = df_points.index
    gdf = pd.merge(gdf, df_points, on='Start_pos', how='inner')

    df_points = pd.DataFrame(s_points, columns=['End_pos'])
    df_points['TNODE_'] = df_points.index
    gdf = pd.merge(gdf, df_points, on='End_pos', how='inner')

    # Bring nodes and their position in form needed for osmnx (give arbitrary osmid (index) despite not osm file)
    df_points.columns = ['pos', 'osmid']
    df_points[['x', 'y']] = df_points['pos'].apply(pd.Series)
    df_node_xy = df_points.drop('pos', axis=1)

    # Create Graph Object
    G = nx.MultiDiGraph(name=name, crs=gdf.crs)

    # Add nodes to graph
    for node, data in df_node_xy.T.to_dict().items():
        G.add_node(node, **data)

    # Add edges to graph
    for i, row in gdf.iterrows():
        dict_row = row.to_dict()
        if 'geometry' in dict_row: del dict_row['geometry']
        G.add_edge(u_for_edge=dict_row['FNODE_'], v_for_edge=dict_row['TNODE_'], **dict_row)

    if make_G_bidi:
        gdf.rename(columns={'Start_pos': 'End_pos',
                            'End_pos': 'Start_pos',
                            'FNODE_': 'TNODE_',
                            'TNODE_': 'FNODE_', }, inplace=True)

        # Add edges to graph
        for i, row in gdf.iterrows():
            dict_row = row.to_dict()
            if 'geometry' in dict_row: del dict_row['geometry']
            G.add_edge(u_for_edge=dict_row['FNODE_'], v_for_edge=dict_row['TNODE_'], **dict_row)

    return G


class RailwayGraph:
    def __init__(
            self,
            routes: str | Path | gp.GeoDataFrame | nx.Graph,
            stations: str | Path | gp.GeoDataFrame | dict,
            directions: list[tuple[str, str]] = None,
    ):
        routes_geo_crs: str = None
        # Загрузим пути жд дороги из shp
        if isinstance(routes, (str, Path)):
            routes = Path(routes)
            # Загрузим GeoDataFrame
            routes: gp.GeoDataFrame = gp.read_file(routes)
        if isinstance(routes, gp.GeoDataFrame):
            # Запомним кодировку точек
            routes_geo_crs = routes.crs
            # Преобразуем в граф:
            routes: gp.GeoDataFrame = convert_gpd2graph(routes, make_G_bidi=True)
        if not isinstance(routes, nx.Graph):
            raise ValueError("'routes' is not file, GeoDataFrame or Graph")
        if isinstance(routes, nx.MultiDiGraph):
            # Удалим параллельные связи между узлами (избыточные)
            routes = ox.convert.to_digraph(routes, weight='length')
            routes = nx.MultiDiGraph(routes)

        # Загрузим станции из shp
        if isinstance(stations, (str, Path)):
            stations = Path(stations)
            # Загрузим GeoDataFrame
            stations: gp.GeoDataFrame = gp.read_file(stations)
        if isinstance(stations, gp.GeoDataFrame):
            # Заменим кодировку точек
            if routes_geo_crs:
                stations = stations.to_crs(routes_geo_crs)
            # Преобразуем станции в словарь
            stations = {
                key: (val.x, val.y) for key, val in zip(stations.name, stations.geometry)
            }
        if not isinstance(stations, dict):
            raise ValueError("'routes' is not file, GeoDataFrame or dict")

        self.routes_graph: nx.Graph = routes
        self._id_2_route_map = list(self.routes_graph)
        self._route_2_id_map = {val: idx for idx, val in enumerate(self._id_2_route_map)}
        self.station_2_point_map: dict[str, tuple[int, int]] = stations

        if directions is not None:
            self._filter_routes(directions)

    def _filter_routes(self, directions: list[tuple[str, str]]):
        unusage_edges_idx = set(range(len(self.routes_graph.edges)))
        for src, dst in tqdm(
                directions,
                desc='Finding routes between stations'
        ):
            route_edges_idx = self.find_route(src, dst)
            unusage_edges_idx -= set(route_edges_idx)

        unusage_edges_point = []

        for i, (u, v, _) in enumerate(self.routes_graph.edges):
            if i in unusage_edges_idx:
                unusage_edges_point.append((u, v))

        self.routes_graph.remove_edges_from(unusage_edges_point)

    def find_route(self, src_station: str, dst_station: str) -> list[int]:
        src_point = self.station_2_point_map[src_station]
        dst_point = self.station_2_point_map[dst_station]

        src_node = ox.nearest_nodes(self.routes_graph, *src_point)
        dst_node = ox.nearest_nodes(self.routes_graph, *dst_point)

        route = ox.shortest_path(self.routes_graph, src_node, dst_node, weight="length")

        return route

    def plot_route(self, route: list[int]):
        fig, ax = ox.plot_graph_route(
            self.routes_graph,
            route,
            route_color="y",
            route_linewidth=6,
            node_size=0
        )
        plt.show()

    def get_route_geodata(self, route: list[int]) -> gp.GeoDataFrame:
        return ox.routing.route_to_gdf(self.routes_graph, route)

    @property
    def stations(self) -> list[str]:
        return list(self.station_2_point_map.keys())

    @property
    def nodes(self):
        return self.routes_graph.nodes

    @property
    def edges(self):
        return self.routes_graph.edges
