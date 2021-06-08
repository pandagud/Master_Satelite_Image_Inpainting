import logging
import click
import random


from src.config_default import TrainingConfig
from src.config_utillity import update_config
import os
import geojson
from geojson import Polygon,FeatureCollection,dump
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from shapely import geometry, affinity
from shapely.ops import unary_union
## TESSTING

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from sentinelsat import InvalidChecksumError
@click.command()
@click.argument('args', nargs=-1)
def main(args):
    """ Runs dataLayer processing scripts to turn raw dataLayer from (../raw) into
        cleaned dataLayer ready to be analyzed (saved in ../processed).
    """
    ## Talk to Rune about how dataLayer is handle. If it should be part of the "big" project.
    ## set number_tiles:1764
    config = TrainingConfig()
    config = update_config(args,config)
    logger = logging.getLogger(__name__)
    logger.info('making final dataLayer set from raw dataLayer')
    userTuple = [['pandagud','damp4ever'],['pandagud2','damp4ever'],['pandagud3','damp4ever'],['au524478','Palantir1234']]
    current_user = random.choice(userTuple)

    api = SentinelAPI(current_user[0], current_user[1], 'https://scihub.copernicus.eu/dhus')

    # search by polygon, time, and SciHub query keywords
    path  = r"C:\Users\panda\Downloads\LC80290292014132LGN00.geojson"
    footprint = geojson_to_wkt(read_geojson(path))
    products = api.query(area=footprint, date=('20210101', '20210105'),
                         platformname='Sentinel-2',order_by='+ingestiondate',limit=1)
    areas = api.to_geodataframe(products)
    geojson = api.to_geojson(products)
    api.download_all(products,into=r'C:\Users\panda\Sat_paper\Alfa')

    products = api.query(area=footprint, date=('20210401', '20210430'), producttype='GRD',
                         platformname='Sentinel-1',sensoroperationalmode='IW',polarisationmode='VV VH',order_by='ingestiondate')
    firstproduct = next(iter(products))
    online_product = ''
    for i in products:
        is_online = api.is_online( products.get(i).get('uuid'))
        if is_online:
            online_product=i
            break
    delete_list = []
    for i in products:
        if i !=online_product:
            delete_list.append(i)
    for i in delete_list:
        del products[i]

    ground_geojsons = read_geojson(path)
    products_geojsons = api.to_geojson(products)

    ground_polygon=ground_geojsons.get('features')[0].get('geometry').get('coordinates')
    ground_polygon = geometry.Polygon(ground_polygon[0][0])
    import numpy as np
    titles = []
    ids = []
    for item in products_geojsons.get('features'):
        id = item.get('properties').get('id')
        item = item.get('properties').get('title')
        item = (item[17:25] + item[48:55])
        titles.append(item)
        ids.append([item,id])
    unique = list(set(titles))
    union_list = []
    for i,element in  enumerate(unique):
        local_polygon = Polygon()
        for j in range(len(titles)):
            if titles[j]==element:
                item = products_geojsons.get('features')[j]
                item = item.get('geometry').get('coordinates')
                item = geometry.Polygon(item[0][0])
                item = affinity.scale(item,xfact=1.01,yfact=1.01)
                polygons = [item,local_polygon]
                local_polygons = unary_union(polygons)
                local_polygon=item
        union_list.append([local_polygons,element])
    for index,element in enumerate(union_list):
        wkt = element[0].wkt
        if ground_polygon.within(element[0]):
            found_id=element[1]
            break
    for i in ids:
        if found_id!=i[0]:
            del products[i[1]]
    area_list=[]
    for index,item in enumerate(products_geojsons.get('features')):
        item=item.get('geometry').get('coordinates')
        item = geometry.Polygon(item[0][0])
        local_intersection = item.intersection(ground_polygon)
        local_intersection = [local_intersection.area,index]
        area_list.append(local_intersection)
    area_list.sort(reverse=True)
    for index in range(len(area_list)):
        item = products_geojsons.get('features')[area_list[index][1]]
        id = item.get('properties').get('id')
        item=item.get('geometry').get('coordinates')
        item = geometry.Polygon(item[0][0])
        if item.intersects(ground_polygon):
            local_intersection = ground_polygon.intersection(item)
            print(str(ground_polygon.area))
            print(str(local_intersection.area))
            # ground_polygon = ground_polygon.difference(local_intersection)
            ground_polygon = (ground_polygon.symmetric_difference(local_intersection)).difference(local_intersection)
        else:
            del products[id]
    import datetime
    from datetime import timedelta
    S2_geojson = read_geojson(path)

    start_S1_date = S2_geojson.get('features')[0].get('properties').get('ingestiondate')
    start_S1_date = start_S1_date.split('T')[0]
    start_S1_date = datetime.datetime.strptime(start_S1_date, '%Y-%m-%d').date()
    ## New end date for S1
    end_S1_date = start_S1_date + timedelta(days=7)
    start_S1_date = start_S1_date -  timedelta(days=7)
    start_S1_date_str=str(start_S1_date).replace('-','')
    end_S1_date_str = str(end_S1_date).replace('-', '')





        ## COMBINE FOOTPRINT
    geom_in_geojson = []
    geom_in_geojson.append(geojson.Feature(geometry=ground_polygon, properties={"MissingData":"Test"}))
    feature_collection = FeatureCollection(geom_in_geojson)
    pathToFile = r'C:\Users\panda\Sat_paper\missing.geojson'
    with open(pathToFile, 'w') as f:
        dump(feature_collection, f)


    print("Done")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()