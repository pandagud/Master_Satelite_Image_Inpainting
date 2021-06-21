import datetime
import os
import duallog
import sentinelsat
import glob

import pandas as pd

from osgeo import gdal
from pathlib import Path
from absl import app, flags, logging
from sentinelsat import read_geojson, geojson_to_wkt

from config import define_flags
from src.downloader import Downloader
from src.processpipeliner import ProcessPipeliner

FLAGS = flags.FLAGS


def main(argv):
    # Setup logging
    duallog.setup(Path(FLAGS.data_directory) / 'logs')
    logger_path = Path(str(FLAGS.end_data_directory))
    logger_path = logger_path.parent.absolute()
    if FLAGS.satellite == 's2':
        files = glob.glob(str(logger_path)+ '/log/*')
        for f in files:
            os.remove(f)
    duallog.setup(logger_path / 'log')
    logging.set_verbosity(FLAGS.logging_verbosity)  # Must be called after duallog.setup() to function properly

    # Configure GDAL
    gdal.SetCacheMax(8*1000000000)
    # Create absolute paths (either use full path provided as argument or use data dir in the project folder)
    data_dir = Path(FLAGS.data_directory) if os.path.isabs(FLAGS.data_directory) else Path.cwd() / FLAGS.data_directory
    print(str(data_dir))
    # Ensure filename on geojson file
    geojson_path = FLAGS.geojson if FLAGS.geojson.endswith('.geojson') else FLAGS.geojson + '.geojson'

    logging.info(str(FLAGS.process_tiles))
    # If no order_id from previous order is provided, then download the data requested for this order
    order_id = FLAGS.order_id
    print(order_id)
    if order_id == 'Empty' or 'test' in order_id:
        if order_id =='Empty':
                order_id = 'order_' + datetime.datetime.today().strftime('%Y%m%d-%H%M%S')

        logging.info("####################################")
        logging.info("# Initializing Sentinel downloader #")
        logging.info("####################################")
        logging.info("Order id: " + order_id)
        downloader = Downloader(username=FLAGS.username, password=FLAGS.password, satellite=FLAGS.satellite,
                                order_id=order_id, directory=data_dir)

        # Load the geojson file (check whether the filename was included in the provided name)
        logging.info(str(geojson_path))
        if 'workspace' in str(geojson_path):
            # Load full path if worksppace
            footprint = geojson_to_wkt(read_geojson(Path(geojson_path)))
        else:
            # Load the provided geojson file from the data directory
            footprint = geojson_to_wkt(read_geojson(data_dir / 'geojson' / geojson_path))  # Load from data directory

        # Query the data (multiple footprints can be used, but it is recommended to stick to a single footprint)
        downloader.query(footprint, FLAGS.startdate, FLAGS.enddate)

        # Following code can be used if several geojson files are to be queried
        # footprint = geojson_to_wkt(read_geojson('data/geojson/bornholm.geojson'))
        # downloader.query(footprint, FLAGS.startdate, FLAGS.enddate)

        # Print the number of products and size of all products to be downloaded
        downloader.print_num_and_size_of_products()
        downloader.save_queried_products()  # Save a geojson containing all products to be downloaded
        downloader.save_queried_products_location(FLAGS.end_data_directory)
        logging.info("")

        if FLAGS.download:
            logging.info("####################")
            logging.info("# Downloading data #")
            logging.info("####################")
            downloader.download_zipfiles()
            logging.info("")
    if FLAGS.process_tiles:
        # Load products to be processed (always load from file to ensure modularity for the downloader and processor)
        queried_products_path = (data_dir / 'orders' / order_id).with_suffix('.pkl')
        products_df = pd.read_pickle(queried_products_path)

        logging.info("###################")
        logging.info("# Processing data #")
        logging.info("###################")
        processpipeliner = ProcessPipeliner(products_df=products_df, directory=data_dir)
        processpipeliner.process_products()


        # preprocessor = PreProcessor(products=products_df, directory=data_dir, overwrite_products=FLAGS.overwrite,
        #                             compress_gtiff=FLAGS.compress, delete_jp2_files=True)
        # preprocessor.preprocess_products()
        #
        # postprocessor = PostProcessor(products=products_df, directory=data_dir,
        #                               overwrite_products=FLAGS.overwrite, compress_gtiff=FLAGS.compress)
        # postprocessor.postprocess_products(ndvi=False, vrt=True, coreg=True)


if __name__ == '__main__':
    define_flags()
    app.run(main)
