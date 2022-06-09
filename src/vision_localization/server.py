import argparse
import logging
import math
import zmq
import flask
from werkzeug.serving import WSGIRequestHandler
import json
from scipy.spatial.transform import Rotation
from threading import Thread, Lock
from math import sqrt
from pathlib import Path
import base64
import numpy as np
from multiprocessing import Process, Queue, Event, Lock
from multiprocessing.shared_memory import SharedMemory
from queue import Full as FullException
import cv2
from .core import *
import time

##### Global variables #####
roaming_markers = dict()
roaming_markers_lock = Lock()
TARGET_MARKER_ID = 4
camera_images = dict()
camera_images_lock = Lock()

##### Utility function #####
def distance(p1, p2):
    return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

##### Flask server #####

# Flask defaults to HTTP 1.0. Use HTTP 1.1 to keep connections alive for high freqeuncy requests.
WSGIRequestHandler.protocol_version = 'HTTP/1.1'

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


##### Main #####

def update_pose(pose_queue:Queue):
    while True:
        poses = pose_queue.get()

        with roaming_markers_lock:
            roaming_markers.update(poses)

        time.sleep(0.001)

def localization_process(config:dict, 
                         camera_name:str,
                         in_shm_name:str,
                         in_event:Event,
                         in_lock:Lock,
                         out_shm_name:str,
                         out_event:Event,
                         out_lock:Lock,
                         pose_queue:Queue):
    # init params
    camera_params = CameraParams.from_config(config, camera_name)
    localization_params = LocalizationParams.from_config(config)
    
    # init shared memory
    in_shm = SharedMemory(name=in_shm_name, create=False)
    in_frame = np.ndarray((camera_params.height, camera_params.width, 3), dtype=np.uint8, buffer=in_shm.buf)
    
    out_shm = SharedMemory(name=out_shm_name, create=False)
    out_frame = np.ndarray((camera_params.height, camera_params.width, 3), dtype=np.uint8, buffer=out_shm.buf)

    # init node
    node = LocalizationNode.from_params(camera_params, localization_params)

    frame = np.ndarray((camera_params.height, camera_params.width, 3), dtype=np.uint8)

    while True:
        in_event.wait()
        with in_lock:
            frame[:] = in_frame[:]
            in_event.clear()
        
        poses, frame = node.process_frame(frame, annotate=True)

        try:
            pose_queue.put_nowait(poses)
        except FullException:
            logging.warning('Pose queue is full!')
            continue
        
        with out_lock:
            out_frame[:] = frame[:] 
            out_event.set() # indicate new frame available

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

    config = load_config(args.config)

    # init ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.subscribe(b'pose')
    socket.subscribe(b'img')

    subprocesses = {}
    in_shm = {}
    in_frames = {}
    in_events = {}
    in_locks = {}
    out_shm = {}
    out_frames = {}
    out_events = {}
    out_locks = {}
    pose_queue = Queue()
    
    for cam_name in config['cameras']:
        cam = config['cameras'][cam_name]
        logging.debug('Connecting to ' + 'tcp://{}:{}'.format(cam['host'], cam['port']))
        socket.connect('tcp://{}:{}'.format(cam['host'], cam['port']))

    if config['process_on_server']:
        for cam_name in config['cameras']:
            # prepare shared memory start subprocesses
            arr = np.ndarray((cam['height'],cam['width'],3), dtype=np.uint8)
            in_shm[cam_name] = SharedMemory(create=True,
                                            size=arr.nbytes)
            in_frames[cam_name] = np.ndarray((cam['height'],cam['width'],3), dtype=np.uint8, buffer=in_shm[cam_name].buf)
            out_shm[cam_name] = SharedMemory(create=True,
                                            size=arr.nbytes)
            out_frames[cam_name] = np.ndarray((cam['height'],cam['width'],3), dtype=np.uint8, buffer=out_shm[cam_name].buf)

            in_events[cam_name] = Event()
            in_locks[cam_name] = Lock()
            out_events[cam_name] = Event()
            out_locks[cam_name] = Lock()

            process = Process(target=localization_process, args=(
                config,
                cam_name,
                in_shm[cam_name].name,
                in_events[cam_name],
                in_locks[cam_name],
                out_shm[cam_name].name,
                out_events[cam_name],
                in_locks[cam_name],
                pose_queue
            ))
            process.start()
            subprocesses[cam_name] = process

        # start thread to update pose
        pose_thread = Thread(target=update_pose, args=(pose_queue,))
        pose_thread.start()

    # start Flask server in separate thread
    server_thread = Thread(target=app.run, args=(config['server']['host'],config['server']['port']), daemon=True)
    server_thread.start()


    while True:
        topic, source, data = socket.recv_multipart()

        if topic == b'pose':
            poses = json.loads(data)
            
            roaming_markers_lock.acquire()
            for marker_id, marker_pose in poses.items():
                marker_id = int(marker_id)
                roaming_markers[marker_id] = marker_pose
            roaming_markers_lock.release()
        elif topic == b'img':
            cam_name = source.decode('utf-8')
            logging.debug('Received image from {}'.format(cam_name))

            with camera_images_lock:
                camera_images[cam_name] = data

            if config['process_on_server']:
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                with in_locks[cam_name]:
                    in_frames[cam_name][:] = frame[:]
                    in_events[cam_name].set()



if __name__ == '__main__':
    main()
