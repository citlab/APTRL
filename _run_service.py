#!/usr/bin/env python


# import ascar.ascar_logging
import daemon
import importlib
from lockfile.pidlockfile import PIDLockFile
import logging
import os
import signal
import sys
import json

__author__ = 'Puriwat Khantiviriya'


def read_conf_file(conf_file):
    with open(conf_file, 'r') as confs:
        conf = json.load(confs)
        return conf

def import_controller_intf(classname, opt, conf):

    # import controller_intf
    classname_part = classname.split('.')
    m = importlib.import_module('.'.join(classname_part[:-1]))
    #import class
    m = getattr(m, classname_part[-1])
    app = m(conf, opt)

    return app

def check_stale_lock(pidfile):
    pidfile_pid = pidfile.read_pid()
    if pidfile_pid is not None:
        try:
            os.kill(pidfile_pid, signal.SIG_DFL)
        except ProcessLookupError as exc:
            # The specified PID does not exist
            pidfile.break_lock()
            return
        print("Process is already running")
        exit(255)
    return

# Get module to run a service
classname = sys.argv[1]
# Reading config from file
conf = read_conf_file(sys.argv[2])
opt = conf['opt']
# Get the process id file
pid_dir = sys.argv[3]

# Adding logging
# logfile = conf[classname + '_logfile'] if classname + '_logfile' in opt.keys()\
#           else '/var/log/{0}.log'.format(classname)
# ascar.add_log_file(logfile, opt.get('log_lazy_flush', False))
# ascar.logger.setLevel(opt['loglevel'] if 'loglevel' in opt else logging.INFO)

# import controller for running daemon
app = import_controller_intf(classname, conf, opt)

pidfile_name = os.path.join(pid_dir if pid_dir != '' else '/var/run',
                            classname + '.pid')

pidfile = PIDLockFile(pidfile_name, timeout=-1)
check_stale_lock(pidfile)
context = daemon.DaemonContext(
    pidfile=pidfile,
    stdout=open(f'log/{classname}_stdout', 'w+'),
    stderr=open(f'log/{classname}_stderr', 'w+'),
)

def stop(signum, frame):
    app.stop()


context.signal_map = {
    signal.SIGTERM: stop,
    signal.SIGHUP: 'terminate',
    # signal.SIGUSR1: reload_program_config,
    }

# context.files_preserve = [ascar.ascar_logging.log_handler.stream]

with context:
    app.start()