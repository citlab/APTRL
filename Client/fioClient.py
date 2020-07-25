import os
import socket
import subprocess

class FioClient:

    def __init__(self, hosts, jobfile):

        self.hosts = hosts
        self.jobfile = jobfile

    def runTest(self):
        # check each host
        cmd = 'fio'
        err = 'Hosts cannot be reached: '
        count = 0
        for host in self.hosts:
            if(self.checkHost(host)):
                cmd += f' --client={host}'
            else:
                count += 1
                err += f'{host}, '
        cmd += f' --remote-config {self.jobfile} --output-format=json'

        # Send cmd to send job tester to hosts(Client that mount ceph storage)
        # receive the output in json format and extract necessary data
        if(count < len(self.hosts)):
            output = subprocess.check_output(cmd, shell=True)
            print(output.decode().strip())
        print(err)

    def checkHost(self, host):
        # send ping command to host
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host,8765))
        if(result == 0):
            return True
        else:
            return False
        sock.close()
        
    def checkVersion(self):
        cmd = 'fio --version'

        output = subprocess.check_output(cmd, shell=True)
        print(f'fio version: {output.decode().strip()}')

f = FioClient(['192.168.1.10','192.168.1.12','192.168.1.27'],  'poisson-rate-submission.fio')
f.checkVersion()
f.runTest()
    