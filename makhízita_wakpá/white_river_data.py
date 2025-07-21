#Retrieve the stream data using the USGS website
from dataretrieval import nwis

def get_white_river_data():
    site_id = "06352000"  # White River near Oglala, SD
    df = nwis.get_record(sites=site_id, service='iv', start='2023-01-01', end='2023-12-31')
    return df

data = get_white_river_data()
print(data.head())
