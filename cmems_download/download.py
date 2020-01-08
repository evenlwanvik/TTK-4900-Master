

import datetime
import os

#pyPath = 'C:/Users/alver/AppData/Local/Programs/Python/Python37/python.exe'
pyPath = 'python'

username = "ewanvik"
password = "Waneve06978!"
storePath = "C:/Users/evenwa/Workspaces/Master/cmems_data"

# Global reanalysis (model):
#dataset = "phys"
#dt = datetime.timedelta(days=1)
#duration = datetime.timedelta(hours=23)
#filePrefix = "phys_"

# Global reprocessed observations:
dataset = "multiobs"
dt = datetime.timedelta(days=7)
duration = datetime.timedelta(days=6)
filePrefix = "multiobs_"

latitude = [45, 90]
longitude = [-60, 60]
startT = datetime.datetime(2018,1,1,0,0,0)

N = 1
time = startT
for i in range(0,N):
    tEnd = time + duration

    n = str(i+1)
    filename = filePrefix+n.zfill(3)+".nc"

    if dataset=="phys":
        command = pyPath+" -m motuclient --motu http://nrt.cmems-du.eu/motu-web/Motu " \
            +"--service-id GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS --product-id global-analysis-forecast-phy-001-024 " \
            +"--longitude-min "+str(longitude[0])+" --longitude-max "+str(longitude[1])+" --latitude-min "+str(latitude[0])+" " \
            "--latitude-max "+str(latitude[1])+" " \
            +"--date-min \""+str(time)+"\" --date-max \""+str(tEnd)+"\""\
            +" --depth-min 0.493 --depth-max 0.4942 --variable thetao --variable bottomT " \
            +"--variable so --variable zos --variable uo --variable vo --variable mlotst --variable siconc " \
            +"--variable sithick --variable usi --variable vsi --out-dir . --out-name "+filename+" " \
            "--user "+username+" --pwd \""+password+"\""
    elif dataset=="multiobs":
        command = pyPath+" -m motuclient --motu http://my.cmems-du.eu/motu-web/Motu " \
            +"--service-id MULTIOBS_GLO_PHY_REP_015_002-TDS --product-id dataset-armor-3d-rep-weekly " \
            +"--longitude-min "+str(longitude[0])+" --longitude-max "+str(longitude[1])+" --latitude-min "+str(latitude[0])+" " \
            "--latitude-max "+str(latitude[1])+" " \
            +"--date-min \""+str(time)+"\" --date-max \""+str(tEnd)+"\""\
            +" --depth-min -1 --depth-max 1 " \
            +"--variable to --variable so --variable zo --variable ugo --variable vgo --variable mlotst " \
            +"--out-dir . --out-name "+storePath+"/"+filename+" " \
            "--user "+username+" --pwd \""+password+"\""

    print(command)
    os.system(command)
    time = time + dt
    print("\n")



