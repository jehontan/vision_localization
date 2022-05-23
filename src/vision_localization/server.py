import argparse
from calendar import c
import logging
import math
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
camera_images = dict()
camera_images_lock = Lock()

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

    R = Rotation.from_quat(pose[-4:]).as_matrix()
    heading = math.degrees(math.atan2(R[1,0], R[0,0]))

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
            'z': heading
        },
        'clues': clues
    }

@app.route('/stream/<cam_name>', methods=['GET'])
def get_cam_stream(cam_name):
    if cam_name not in camera_images.keys():
        return 'Camera not configured to stream.', 404
    
    return (
        '<html><head><title>{0} Stream</title></head><body><img src="/mjpg/{0}"/></body></html>'.format(cam_name)
    )
    
@app.route('/mjpg/<cam_name>', methods=['GET'])
def get_cam_mjpg(cam_name):
    return flask.Response(mjpg_generator(cam_name), mimetype='multipart/x-mixed-replace; boundary=fram')

def mjpg_generator(cam_name):
    while True:
        camera_images_lock.acquire()
        frame = camera_images[cam_name]
        camera_images_lock.release()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


##### Configuration loading #####

def load_config(args):
    '''Load markers from config YAML file.'''
    global config, host, port, roaming_markers, camera_images, TARGET_MARKER_ID

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    host = config['server']['host']
    port = config['server']['port']

    for marker in config['roaming_markers']:
        roaming_markers[marker['id']] = None

    for cam_name in config['cameras']:
        if config['cameras'][cam_name]['stream']:
            camera_images[cam_name] = None

    TARGET_MARKER_ID = config['roaming_markers'][0]['id']

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
    socket.subscribe(b'img')
    
    for cam_name in config['cameras']:
        cam = config['cameras'][cam_name]
        logging.debug('Connecting to ' + 'tcp://{}:{}'.format(cam['host'], cam['port']))
        socket.connect('tcp://{}:{}'.format(cam['host'], cam['port']))

    # start Flask server
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()

    while True:
        buf = socket.recv_multipart()
        topic = buf[0]

        if topic == b'pose':
            poses = json.loads(buf[1])
            
            roaming_markers_lock.acquire()
            for marker_id, marker_pose in poses.items():
                marker_id = int(marker_id)
                roaming_markers[marker_id] = marker_pose
            roaming_markers_lock.release()
        if topic == b'img':
            cam_name = buf[1].decode('utf-8')
            camera_images_lock.acquire()
            camera_images[cam_name] = buf[2]
            camera_images_lock.release()


if __name__ == '__main__':
    main()