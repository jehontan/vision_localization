import cv2
import argparse
import yaml
import logging
from scipy.spatial.transform import Rotation
import numpy as np
import zmq
import json
from enum import IntEnum

##### Global variables #####
config = dict()             # global configuration
fixed_markers = dict()      # keys: marker id, value: origin -> marker transformation
roaming_markers = list()    # list of roaming marker ids to watch
K = np.zeros((3,3))         # Camera matrix
D = D = np.zeros((1,5))     # Distortion parameters
host = '0.0.0.0'
port = 5912
marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
is_streaming = False

##### Utility functions #####

def transformation(rvec, tvec):
    '''Compose transformation martix from rotation and translation vectors.'''
    T = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T[:3,:3] = R
    T[:3,3] = tvec
    return T

def rvec2quat(rvec):
    '''Convert rottation vector to quaternion.'''
    angle = np.linalg.norm(rvec)
    return np.hstack([rvec/angle*np.sin(angle/2), np.cos(angle/2)])

##### Configuration loading #####

def load_config(args):
    '''Load markers from config YAML file.'''
    global config, fixed_markers, roaming_markers, host, port, K, D, is_streaming

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    cam = config['cameras'][args.camera]

    host = cam['host']
    port = cam['port']
    fx = cam['fx']
    fy = cam['fy']
    cx = cam['cx']
    cy = cam['cy']
    
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    D = np.array(cam['D'])

    is_streaming = cam['stream']

    for marker in config['fixed_markers']:
        position = marker['position']
        orientation = marker['orientation']
        R = Rotation.from_euler('xyz', [orientation['x'], orientation['y'], orientation['z']])
        t = -np.array([position['x'], position['y'], position['z']])
        T = np.eye(4)
        T[:3,:3] = R.as_matrix()
        T[:3,3] = t
        fixed_markers[marker['id']] = T

    for marker in config['roaming_markers']:
        roaming_markers.append(marker['id'])

##### Node class ####

class SelectPolicy(IntEnum):
    FIRST = 0
    CLOSEST = 1
    BEST = 2

class LocalizationNode:
    def __init__(self,
                 marker_dict,
                 marker_size,
                 K,
                 D,
                 fixed_markers,
                 roaming_markers,
                 name='LocalizationNode',
                 select_policy='first'):

        self.marker_dict = marker_dict
        self.marker_size = marker_size
        self.K = K
        self.D = D
        self.fixed_markers = fixed_markers
        self.roaming_markers = roaming_markers
        self.name = name
        self.logger = logging.getLogger(name)

        map_select_policy = {
            'first': SelectPolicy.FIRST,
            'closest': SelectPolicy.CLOSEST,
            'best': SelectPolicy.BEST,
        }
        self.select_policy = map_select_policy[select_policy]

    def frame_callback(self, frame):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.marker_dict)

        if self.select_policy == SelectPolicy.CLOSEST:
            centers = [np.average(c[0], axis=0) for c in corners]
        elif self.select_policy == SelectPolicy.BEST:
            quality = [abs((c[0][0,:]-c[0][2:0])/(c[0][1,:]-c[0][3,:])) for c in corners] 

        roaming_poses = dict()

        if not corners:
            # no markers, early terminate
            self.logger.warn('No markers found.')
        else:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.K, self.D)
            ids_ = ids.flatten().tolist()
            
            fixed_inds = [] # indices corresponding to observed fixed markers
            roam_inds = []  # indices corresponding to observed roaming markers

            for i, id_ in enumerate(ids_):
                if id_ in self.fixed_markers.keys():
                    fixed_inds.append(i)
                
                if id_ in self.roaming_markers:
                    roam_inds.append(i)

            if not fixed_inds:
                self.logger.warn('No fixed markers seen.')
            else:
                # iterate over observed roaming markers
                
                if self.select_policy == SelectPolicy.CLOSEST:
                    fixed_centers = [centers[f] for f in fixed_inds]
                elif self.select_policy == SelectPolicy.BEST:
                    fixed_quality = [quality[f] for f in fixed_inds]

                for roam_ind in roam_inds:
                    # pick fixed marker reference based on policy
                    if self.select_policy == SelectPolicy.FIRST:
                        # default to picking first seen fixed marker
                        fixed_ind = fixed_inds[0]
                    elif self.select_policy == SelectPolicy.CLOSEST:
                        # pick closest marker
                        distances = np.linalg.norm(np.subtract(fixed_centers, centers[roam_ind][0]), axis=1)
                        fixed_ind = fixed_inds[np.argmin(distances)]
                    elif self.select_policy == SelectPolicy.BEST:
                        # use best marker                            
                        fixed_ind = fixed_inds[np.argmax(fixed_quality)]

                    # calculate transform
                    origin_to_fixed = self.fixed_markers[ids_[fixed_ind]]
                    fixed_to_cam = transformation(rvecs[fixed_ind], tvecs[fixed_ind])
                    origin_to_cam = fixed_to_cam @ origin_to_fixed

                    robot_to_cam = transformation(rvecs[roam_ind], tvecs[roam_ind])

                    robot_to_origin = np.linalg.inv(origin_to_cam)@robot_to_cam

                    position = robot_to_origin[:,3]
                    
                    rot = Rotation.from_matrix(robot_to_origin[:3,:3])
                    quat = rot.as_quat()

                    roaming_poses[ids_[roam_ind]] = [
                        position[0],
                        position[1],
                        position[2],
                        quat[0],
                        quat[1],
                        quat[2],
                        quat[3]
                    ]

        return roaming_poses, ids, corners


##### Main function #####

def main():
    global config, fixed_markers, roaming_markers, marker_dict, host, port, K, D, is_streaming

    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Configuration file.')
    parser.add_argument('camera', type=str, help='Camera name.')
    parser.add_argument('-i', dest='camera_num', required=False, type=int, default=0, help='Camera input number. Default: 0')
    parser.add_argument('--policy', dest='policy', required=False, type=str, default='first', help='Fixed marker selection policy. [first|closest|best]. Default: first')
    parser.add_argument('--log', dest='log_level', type=str, default='info', required=False, help='Logging level. [debug|info|warn|error|critical]. Default: info')

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

    # init camera
    camera = cv2.VideoCapture(args.camera_num)

    # init ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind('tcp://{}:{}'.format(host, port))

    # init node
    node = LocalizationNode(marker_dict=marker_dict,
                            marker_size=config['marker_size'],
                            K = K,
                            D = D,
                            fixed_markers=fixed_markers,
                            roaming_markers=roaming_markers,
                            select_policy=args.policy)

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                logging.warn('Cannot get frame from camera.')
                continue
            
            roaming_poses, corners, ids = node.frame_callback(frame)

            # send the poses
            buf = json.dumps(roaming_poses).encode('utf-8')
            socket.send_multipart([b'pose', buf])
            logging.debug('sent')

            # stream frame
            if is_streaming:
                
                # annotate frame
                if corners:
                    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                data = cv2.imencode('.jpg', frame)[1].tobytes()
                socket.send_multipart([b'img', args.camera.encode('utf-8'), data])
                logging.debug('Sent image.')
    finally:
        camera.release()

if __name__ == '__main__':
    main()