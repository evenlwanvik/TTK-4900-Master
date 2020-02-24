import datetime
import os
import yaml

#pyPath = 'C:/Users/alver/AppData/Local/Programs/Python/Python37/python.exe'
pyPath = 'python'


# Open locally stored credentials
conf = yaml.load(open('../../config/credentials.yml'))
username = conf['CMEMS-download']['credentials']['username']
password = conf['CMEMS-download']['credentials']['password']

# Choose directory
storePath = "C:/Master/data/cmems_data/global_10km/"
if not os.path.exists(storePath):
    os.makedirs(storePath)

# =========================================================
# ======= GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS ========
# =========================================================

# Choose service and product id
serviceId = "GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS"
# In below product ID the variables are merged, while in "global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh" the data is split.
productId = "global-analysis-forecast-phy-001-024"
variables = "--variable thetao --variable so --variable uo --variable vo --variable zos" 
#productId = "global-analysis-forecast-phy-001-024-statics"
#variables = "--variable deptho --variable mask" 

# =========================================================
# ======= GLOBAL_REANALYSIS_PHY_001_030-TDS ========
# =========================================================
"""
# Choose service and product id
serviceId = "GLOBAL_REANALYSIS_PHY_001_030-TDS"
# In below product ID the variables are merged, while in "global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh" the data is split.
productId = "global-reanalysis-phy-001-030-daily"
variables = "--variable thetao --variable so --variable uo --variable vo --variable zos --variable mlotst" 
"""
# =========================================================
# ======= GLOBAL_ANALYSIS_FORECAST_WAV_001_027-TDS ========
# =========================================================
"""
# Choose service and product id
serviceId = "GLOBAL_ANALYSIS_FORECAST_WAV_001_027-TDS"
# In below product ID the variables are merged, while in "global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh" the data is split.
productId = "global-reanalysis-phy-001-030-daily-statics"
variables = "--variable deptho" 
"""
# =========================================================
# =================== Date and lon/lat ====================
# =========================================================

dt = datetime.timedelta(days=10)
duration = datetime.timedelta(days=200)
# Global reanalysis (model):

# Physics
filePrefix = "phys_noland_002"
# Bathmetry or statics
#filePrefix = "bathmetry_"
# Global reprocessed observations:
#filePrefix = "multiobs_"

latitude = [45, 90]
longitude = [-60, 60]
# lat/lon far from land for testing OW algorithm
latitude = [45, 60]
longitude = [-42, -15]

startT = datetime.datetime(2016,1,1,0,0,0) + datetime.timedelta(days=0)

N = 1
time = startT
for i in range(0,N):
    tEnd = time + duration

    n = str(i+1)
    filename = filePrefix+n.zfill(3)+".nc"
    if filePrefix in ("phys_", "phys_noland_", "phys_noland_002"):
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