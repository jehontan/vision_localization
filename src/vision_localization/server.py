import argparse
import logging
import yaml
import zmq
import flask
import json
from scipy.spatial.transform import Rotation
from threading import Thread, Lock
from math import sqrt
from pathlib import Path
import base64

##### Global variables #####

config = dict()
host = '0.0.0.0'
port = 5566
roaming_markers = dict()
roaming_markers_lock = Lock()
TARGET_MARKER_ID = 4

##### Utility function #####
def distance(p1, p2):
    return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

##### Flask server #####

app = flask.Flask(__name__)

@app.route('/pose', methods=['GET'])
def get_pose():
    global TARGET_MARKER_ID, roaming_markers, roaming_markers_lock

    if TARGET_MARKER_ID not in roaming_markers.keys():
        logging.warn('Roaming marker not in configuration.')
        return 'Roaming marker not in configuration.', 500

    roaming_markers_lock.acquire()
    pose = roaming_markers[TARGET_MARKER_ID]
    roaming_markers_lock.release()
    
    if pose is None:
        logging.warn('Roaming marker not yet observed.')
        return 'Roaming marker not yet observed.', 404

    R = Rotation.from_quat(pose[-4:]).as_euler('zyx', degrees=True)

    # get clues
    clues = []
    for clue in config['clues']:
        trigger = clue['trigger']
        if distance(pose[:2], (trigger['x'], trigger['y'])) <= trigger['r']:
            audio_bytes = Path(clue['audio_file']).read_bytes()

            clues.append({
                'clue_id': clue['clue_id'],
                'location': clue['location'],
                'audio': base64.encodebytes(audio_bytes).decode('utf-8')
            })

    return {
        'pose': {
            'x': float(pose[0]),
            'y': float(pose[1]),
            'z': float(R[0])
        },
        'clues': clues
    }


##### Configuration loading #####

def load_config(args):
    '''Load markers from config YAML file.'''
    global config, host, port, roaming_markers

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    host = config['server']['host']
    port = config['server']['port']

    for marker in config['roaming_markers']:
        roaming_markers[marker['id']] = None

def start_server():
    global host, port
    app.run(host, port)

def main():
    global config, roaming_markers, roaming_markers_lock

    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Configuration file.')
    parser.add_argument('--log', dest='log_level', type=str, default='info', required=False, help='Logging level. [debug|info|warn|error|critical]')

    args = parser.parse_args()

    # setup logging
    map_log_level = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    logging.basicConfig(level=map_log_level[args.log_level],
                format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                datefmt='%H:%M:%S')

    load_config(args)

    # init ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.subscribe(b'pose')
    
    for cam_name in config['cameras']:
        cam = config['cameras'][cam_name]
        logging.debug('Connecting to ' + 'tcp://{}:{}'.format(cam['host'], cam['port']))
        socket.connect('tcp://{}:{}'.format(cam['host'], cam['port']))

    # start Flask server
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()

    while True:
        _, data = socket.recv_multipart()
        poses = json.loads(data)
        
        roaming_markers_lock.acquire()
        for marker_id, marker_pose in poses.items():
            marker_id = int(marker_id)
            roaming_markers[marker_id] = marker_pose
        roaming_markers_lock.release()

if __name__ == '__main__':
    main()