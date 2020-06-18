import datetime
import os
import yaml

def download_nc(longitude, latitude):
    """ Script for downloading NetCDF files from CMEMS 
    Define input lon and lat boundaries in degrees, e.g., longitude=[45, 60]"""

    pyPath = 'python'

    # Open locally stored credentials, optionally hardcode username and password 
    # (because this is on Github, I use credentials stored on my PC)
    conf = yaml.load(open('C:/Users/47415/master/TTK-4900-Master/config/credentials.yml'), yaml.FullLoader)
    username = conf['CMEMS-download']['credentials']['username']
    password = conf['CMEMS-download']['credentials']['password']

    # Choose directory
    #storePath = "D:/Master/data/cmems_data/global_10km/full/"
    storePath = "D:/Master/data/compare/satellite/"
    #storePath = "D:/Master/data/cmems_data/global_10km/noland/realtime/"
    if not os.path.exists(storePath):
        os.makedirs(storePath)


    # =========================================================
    # ======= GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS ========  MAIN ONE
    # =========================================================
   
    # Choose service and product id
    serviceId = "GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS"
    # In below product ID the variables are merged, while in "global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh" the data is split.
    productId = "global-analysis-forecast-phy-001-024"
    variables = "--variable thetao --variable so --variable uo --variable vo --variable zos" 
    #productId = "global-analysis-forecast-phy-001-024-statics"
    #variables = "--variable deptho --variable mask" 
   
    # =========================================================
    # ======= MULTIOBS_GLO_PHY_NRT_015_001 ========
    # =========================================================
    """
    storePath = "D:/Master/data/compare/satellite2/"
    # Choose service and product id
    serviceId = "MULTIOBS_GLO_PHY_NRT_015_001-TDS"
    # In below product ID the variables are merged, while in "global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh" the data is split.
    productId = "dataset-armor-3d-nrt-weekly"
    variables = "--variable to --variable so --variable ugo --variable vgo --variable zo" 
    """


    # =========================================================
    # =================== Date and lon/lat ====================
    # =========================================================

    dt = datetime.timedelta(days=1)
    duration = datetime.timedelta(days=1)
    # Global reanalysis (model):

    # Physics
    filePrefix = "phys_"
    # Bathmetry or statics
    #filePrefix = "bathmetry_"
    # Global reprocessed observations:
    #filePrefix = "multiobs_"

    #N = 365 # Number of days, or dt
    N = 1
    time = datetime.datetime(2017,6,1,0,0,0) + datetime.timedelta(days=0)
    for i in range(0,N):
        tEnd = time + duration

        n = str(i+100)
        filename = filePrefix+n.zfill(3)+".nc"
        if filePrefix in ("phys_", "phys_noland_2018_", "phys_noland_2016_"):
            command = pyPath+" -m motuclient --motu http://nrt.cmems-du.eu/motu-web/Motu " \
                +"--service-id "+serviceId+" --product-id "+productId+" " \
                +"--longitude-min "+str(longitude[0])+" --longitude-max "+str(longitude[1])+" --latitude-min "+str(latitude[0])+" " \
                "--latitude-max "+str(latitude[1])+" " \
                +"--date-min \""+str(time)+"\" --date-max \""+str(tEnd)+"\""\
                +" --depth-min -1 --depth-max 1 " \
                +variables+ " " \
                +"--out-dir . --out-name "+storePath+"/"+filename+" " \
                +"--user "+username+" --pwd \""+password+"\""

        elif filePrefix=="multiobs_":
            command = pyPath+" -m motuclient --motu http://my.cmems-du.eu/motu-web/Motu " \
                +"--service-id MULTIOBS_GLO_PHY_REP_015_002-TDS --product-id dataset-armor-3d-rep-weekly " \
                +"--longitude-min "+str(longitude[0])+" --longitude-max "+str(longitude[1])+" --latitude-min "+str(latitude[0])+" " \
                "--latitude-max "+str(latitude[1])+" " \
                +"--date-min \""+str(time)+"\" --date-max \""+str(tEnd)+"\""\
                +" --depth-min -1 --depth-max 1 " \
                +"--variable to --variable so --variable zo --variable ugo --variable vgo --variable mlotst " \
                +"--out-dir . --out-name "+storePath+"/"+filename+" " \
                "--user "+username+" --pwd \""+password+"\""

        elif filePrefix=="bathmetry_":
            command = pyPath+" -m motuclient --motu http://nrt.cmems-du.eu/motu-web/Motu " \
                +"--service-id "+serviceId+" --product-id "+productId+" " \
                +"--longitude-min "+str(longitude[0])+" --longitude-max "+str(longitude[1])+" --latitude-min "+str(latitude[0])+" " \
                "--latitude-max "+str(latitude[1])+" " \
                +"--date-min \""+str(time)+"\" --date-max \""+str(tEnd)+"\""\
                +variables+ " " \
                +"--out-dir . --out-name "+storePath+"/"+filename+" " \
                +"--user "+username+" --pwd \""+password+"\""

        print(command)
        os.system(command)
        time = time + dt
        print("\n")


if __name__ == '__main__':
    latitude = [45, 90]
    longitude = [-60, 60]
    # lat/lon far from land for testing OW algorithm
    #latitude = [45, 60]
    #longitude = [-42, -15]
    #latitude = [45, 50]
    #longitude = [-12, -0]
    #latitude = [50, 52]
    #longitude = [-19.6, -17.0]
    #latitude = [45, 50]
    #longitude = [-24, -12]
    download_nc(longitude, latitude) 