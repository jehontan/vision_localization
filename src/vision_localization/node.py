import cv2
import argparse
import logging
import zmq
import json
from .core import *

##### Main function #####

def main():
    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Configuration file.')
    parser.add_argument('camera', type=str, help='Camera name.')
    parser.add_argument('-i', dest='camera_num', required=False, type=int, default=0, help='Camera input number. Default: 0')
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

    # get config
    config = load_config(args.config)
    cam_name_encoded = args.camera.encode('utf-8')

    # init camera
    camera = cv2.VideoCapture(args.camera_num)

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.bind('tcp://{}:{}'.format(config['cameras'][args.camera]['host'], config['cameras'][args.camera]['port']))

    try:
        if config['process_on_server']:
            # only stream images
            while True:
                ret, frame = camera.read()
                _, buf = cv2.imencode('.png', frame)
                socket.send_multipart([b'img', cam_name_encoded, buf.tobytes()])
                logging.debug('Sent image')

        else:
            camera_params = CameraParams.from_config(config, args.camera)
            localization_params = LocalizationParams.from_config(config)
            node = LocalizationNode.from_params(camera_params, localization_params)

            while True:
                ret, frame = camera.read()
                poses, frame = node.process_frame(frame, annotate=True)

                socket.send_multipart([b'pose', cam_name_encoded, json.dumps(poses).encode('utf-8')])

                _, buf = cv2.imencode('.png', frame)
                socket.send_multipart([b'img', cam_name_encoded, buf.tobytes()])
                
    finally:
        camera.release()
        socket.close()


if __name__ == '__main__':
    main()
