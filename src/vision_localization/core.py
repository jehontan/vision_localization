import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, Any
from dataclasses import dataclass
from enum import IntEnum
import logging
import cv2
from multiprocessing import Queue, Event, Lock
from multiprocessing.shared_memory import SharedMemory
from queue import Full as FullException

##### Globals #####
MARKER_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)


def transformation(rvec, tvec):
    '''
    Compose transformation martix from rotation and translation vectors.
    
    Parameters
    ----------
    rvec
        Rotation vector.
    tvec
        Translation vector.

    Returns
    -------
    transformation
        Numpy 4x4 transformation matrix.
    '''
    T = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T[:3,:3] = R
    T[:3,3] = tvec
    return T

def rvec2quat(rvec):
    '''Convert rottation vector to quaternion.
    
    Parameters
    ----------
    rvec
        Rotation vector.
    
    Returns
    -------
    quat
        Quaternion in numpy vector (1x4)
    '''
    angle = np.linalg.norm(rvec)
    return np.hstack([rvec/angle*np.sin(angle/2), np.cos(angle/2)])

def load_config(fname:str) -> dict:
    '''
    Load configuration from YAML file.

    Parameters
    ----------
    fname : str
        Filename of the YAML file
    
    Returns
    -------
    config : dict
        Configuration dictionary.
    '''
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)

    config['select_policy'] = SelectPolicy.from_str(config['select_policy'])

    for camera_name in config['cameras']:
        camera = config['cameras'][camera_name]
        fx = camera['fx']
        fy = camera['fy']
        cx = camera['cx']
        cy = camera['cy']
        camera['K'] = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
        camera['D'] = np.array(camera['D'])

    fixed_markers = {}
    roaming_markers = []

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

    config['fixed_markers'] = fixed_markers
    config['roaming_markers'] = roaming_markers

    return config

class SelectPolicy(IntEnum):
    FIRST = 0
    CLOSEST = 1
    BEST = 2

    @staticmethod
    def from_str(astr:str):
        _map = {
            'first': SelectPolicy.FIRST,
            'closest': SelectPolicy.CLOSEST,
            'nearest': SelectPolicy.CLOSEST,
            'best': SelectPolicy.BEST
        }
        return _map[astr]

@dataclass
class CameraParams:
    camera_name: str
    K: Any
    D: Any
    width: int
    height: int

    @staticmethod
    def from_config(config:dict, camera_name:str):
        cam = config['cameras'][camera_name]
        params = CameraParams(
            camera_name=camera_name,
            K=cam['K'],
            D=cam['D'],
            width=cam['width'],
            height=cam['height'],
        )

        return params

@dataclass
class LocalizationParams:
    marker_dict: Any
    marker_size: float
    fixed_markers: dict
    roaming_markers: list
    select_policy: int

    @staticmethod
    def from_config(config:dict):
        params = LocalizationParams(
            marker_dict=MARKER_DICT,
            marker_size=config['marker_size'],
            fixed_markers=config['fixed_markers'],
            roaming_markers=config['roaming_markers'],
            select_policy=config['select_policy']
        )

        return params

class LocalizationNode:
    def __init__(self,
                 K,
                 D,
                 marker_dict,
                 marker_size,
                 fixed_markers,
                 roaming_markers,
                 select_policy='first',
                 name='LocalizationNode',):

        self.K = K
        self.D = D
        self.marker_dict = marker_dict
        self.marker_size = marker_size
        self.fixed_markers = fixed_markers
        self.roaming_markers = roaming_markers
        self.name = name
        self.logger = logging.getLogger(name)
        self.select_policy = select_policy

    @staticmethod
    def from_params(camera_params:CameraParams, localization_params:LocalizationParams):
        return LocalizationNode(
            K=camera_params.K,
            D=camera_params.D,
            marker_dict=localization_params.marker_dict,
            marker_size=localization_params.marker_size,
            fixed_markers=localization_params.fixed_markers,
            roaming_markers=localization_params.marker_size,
            select_policy=localization_params.select_policy,
            name=camera_params.camera_name
        )

    def process_frame(self, frame, annotate=False):
        '''
        Process image frame.

        Parameters
        ----------
        frame
            OpenCV image to process.
        annotate : bool
            Annotate frame.

        Returns
        -------
        roaming_poses : dict
            Map of roaming marker id to correseponding pose.
        frame
            Frame with annotations is specified.
        '''

        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.marker_dict)

        if self.select_policy == SelectPolicy.CLOSEST:
            centers = [np.average(c[0], axis=0) for c in corners]
        elif self.select_policy == SelectPolicy.BEST:
            quality = list()
            for c in corners:
                d1 = np.linalg.norm(c[0][0,:]-c[0][2,:])
                d2 = np.linalg.norm(c[0][1,:]-c[0][3,:])
                quality.append(d1/d2 if d1 <= d2 else d2/d1)

        roaming_poses = dict()

        if not corners:
            # no markers, early terminate
            self.logger.warn('No markers found.')
        else:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.K, self.D)
            ids_ = ids.flatten().tolist()
            
            fixed_inds = [] # indices corresponding to observed fixed markers
            roaming_inds = []  # indices corresponding to observed roaming markers
            selected_inds = set() # indices corresponding to selected fixed markers

            for i, id_ in enumerate(ids_):
                if id_ in self.fixed_markers.keys():
                    fixed_inds.append(i)
                
                if id_ in self.roaming_markers:
                    roaming_inds.append(i)

            if not fixed_inds:
                self.logger.warn('No fixed markers seen.')
            else:
                # compute metrics for selection policy
                if self.select_policy == SelectPolicy.CLOSEST:
                    fixed_centers = [centers[f] for f in fixed_inds]
                elif self.select_policy == SelectPolicy.BEST:
                    fixed_quality = [quality[f] for f in fixed_inds]

                # iterate over observed roaming markers
                for roam_ind in roaming_inds:
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

                    selected_inds.add(fixed_ind)

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

                if annotate:
                    selected_ids = [ids[i] for i in selected_inds]
                    selected_corners = [corners[i] for i in selected_inds]

                    roaming_ids = [ids[i] for i in roaming_inds]
                    roaming_corners = [corners[i] for i in roaming_inds]

                    frame = cv2.aruco.drawDetectedMarkers(frame, selected_corners, selected_ids, borderColor=(0,255,0))
                    frame = cv2.aruco.drawDetectedMarkers(frame, roaming_corners, roaming_ids, borderColor=(0,0,255))

            return roaming_poses, frame
