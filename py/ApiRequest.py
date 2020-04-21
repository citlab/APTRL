import requests as req
import json

cert_file = 'ceph-dashboard.crt'

class ApiRequest:


    def __init__(self, addr, port):
        self.addr = addr
        self.port = port
        self.header = {
            'Content-Type': 'application/json'
        }

    """
        get Auth token from dashboard login 
        :param :usr: username for Dashboard :passd: password for Dashboard 
        :return: (Signed into Dashboard and add token to header)
    """
    def auth(self, usr, passd):

        path = "/api/auth"
        url = "https://"+self.addr+':'+self.port+path
        payload = "{\"username\":\""+usr+"\", \"password\":\""+passd+"\"}"
        r = req.post(url, verify=cert_file, headers=self.header, data=payload)

        if(r.status_code == req.codes.created):
            token = r.json()['token']
            self.header['Authorization'] = 'Bearer ' + token
        else:
            raise r.raise_for_status()


    """
        Get all API Path from Dashboard module
        :param 
        :return: list of available path in Dashboard API
    """
    def paths(self):

        path = "/docs/api.json"
        url = "https://"+self.addr+':'+self.port+path
        r = req.get(url, verify=cert_file, headers=self.header)

        if(r.status_code == req.codes.ok):
            docs = r.json()
            self.paths = docs['paths']
            return list(docs['paths'].keys())
        else:
            raise r.raise_for_status()


    """
        GET: get all cluster configuration
        POST: update parameter value
        :param :param: name of the parameter that is selected to update value :val: update value of parameter
        :return: GET: list of all cluster config
                 POST: null
    """
    def clusterConfig(self, param=None, val=None, section=None):

        path='/api/cluster_conf'
        url = "https://"+self.addr+':'+self.port+path
        if(val == None):
            if(param == None):
                r = req.get(url, verify=cert_file, headers=self.header)
            else:
                r = req.get(url+"/"+param, verify=cert_file, headers=self.header)
        else:
            if(param == None or section == None):
                raise Exception("No Parameter or Section name provided")
            payload = "{\"name\": \""+param+"\", \"value\": [{\"section\": \""+section+"\",\"value\": \""+val+"\"}]}"
            print(payload)
            r = req.request('POST', url, verify=cert_file, 
            headers=self.header, data=payload)
        if(r.status_code == req.codes.ok):
            config = r.json()
            return config
        elif(r.status_code == req.codes.created):
            return
        else:
            raise r.raise_for_status()

    """
        get health report in (full or minimal)
        :param :report: type of health report in 'full' or 'minimal'
        :return: list of health report
    """
    def health(self, report='full'):

        path='/api/health/'+report
        url = 'https://'+self.addr+':'+self.port+path
        r = req.get(url, verify=cert_file, headers=self.header)

        if(r.status_code == req.codes.ok):
            health = r.json()
            return health
        else:
            raise r.raise_for_status()

    """
        get performance report of mon, osd, mds, ...
        :param :section: section of ceph that you wants to check performance 
               :id: id of selected section
        :return: list of performance report
    """
    def performance(self, section=None, sectionId=None):

        if(section == None or sectionId == None):
            path='/api/perf_counters'
        else:
            path='/api/perf_counters/'+section+'/'+sectionId

        url = 'https://'+self.addr+':'+self.port+path
        r = req.get(url, verify=cert_file, headers=self.header)

        if(r.status_code == req.codes.ok):
            health = r.json()
            return health
        else:
            raise r.raise_for_status()


a = ApiRequest("ceph-dashboard","41716")
allPath = a.paths()
# print(allPath)
a.auth('admin', 'admin')
a.clusterConfig(param='osd_recovery_sleep', val='0', section='global')
#print(a.clusterConfig(param='osd_recovery_sleep'))

#print(a.health(report='minimal'))
print(a.performance('mon','a'))