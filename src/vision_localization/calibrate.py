import time
import cv2
import numpy as np
import logging
import argparse
import flask
from threading import Thread, Lock

marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
global_frame = None
global_frame_lock = Lock()

app = flask.Flask(__name__)

@app.route('/', methods=['GET'])
def get_cam_stream():
    return (
        '<html><head><title>Preview Stream</title></head><body><img src="/preview"/></body></html>'
    )

@app.route('/preview', methods=['GET'])
def get_cam_mjpg():
    return flask.Response(mjpg_generator(), mimetype='multipart/x-mixed-replace; boundary=fram')

def mjpg_generator():
    global global_frame, global_frame_lock
    while True:
        global_frame_lock.acquire()
        local_frame = global_frame
        global_frame_lock.release()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + local_frame + b'\r\n')

def start_server():
    global args
    app.run(args.host, args.port)

def main():
    global marker_dict, global_frame, global_frame_lock, args

    parser = argparse.ArgumentParser(description='Camera calibration program.')
    parser.add_argument('-n', dest='num_images', required=False, default=100, type=int, help='Number of images to collect for calibration. Default: 100.')
    parser.add_argument('-v', dest='visualize', action='store_true', help='Show image preview to help positioning.')
    parser.add_argument('-a', dest='host', required=False, type=str, default='0.0.0.0', help='Visualization server host. Default: "0.0.0.0".')
    parser.add_argument('-p', dest='port', required=False, type=int, default=5127, help='Visualization server port. Default: 5127.')
    parser.add_argument('--log', dest='log_level', type=str, default='info', required=False, help='Logging level. [debug|info|warn|error|critical]')
    parser.add_argument('-o', dest='out_file', required=False, type=str, default='calibOutput.txt', help='Output file path. Default: "./calibOutput.txt"')
    parser.add_argument('-rows', dest='board_rows', required=False, type=int, default=8, help='CharUco board number of rows. Default: 8.')
    parser.add_argument('-cols', dest='board_cols', required=False, type=int, default=11, help='CharUco board number of columns. Default: 11.')
    parser.add_argument('-marker', dest='marker_size', required=False, type=float, default=0.012, help='CharUco marker size. Default: 0.012.')
    parser.add_argument('-chess', dest='chess_size', required=False, type=float, default=0.015, help='CharUco marker size. Default: 0.015.')

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


    # setup ArUco board
    board = cv2.aruco.CharucoBoard_create(args.board_cols, args.board_rows, args.chess_size, args.marker_size, marker_dict)

    allCharucoCorners = []
    allCharucoIds = []

    camera = cv2.VideoCapture(0)
    _, frame = camera.read()
    img_size = frame.shape[:2]

    # start server
    server_thread = Thread(target=start_server, daemon=True)
    if args.visualize:
        server_thread.start()
        logging.debug('Visualization server started.')

    # collect images
    while len(allCharucoIds) < args.num_images:
        time.sleep(0.1)

        ret, frame = camera.read()

        if not ret:
            logging.warn('Bad frame')
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, marker_dict)

            if not corners:
                logging.warn('Board not detected.')
            else:
                ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)

                if ret >= 6:
                    allCharucoCorners.append(charucoCorners)
                    allCharucoIds.append(charucoIds)
                    logging.info('{} of {}'.format(len(allCharucoIds), args.num_images))

                    frame = cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, cornerColor=(0,0,255))

            if args.visualize:
                with global_frame_lock:
                    global_frame = cv2.imencode('.jpg',frame)[1].tobytes()

    repError, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCharucoCorners,
                                                                    allCharucoIds,
                                                                    board,
                                                                    img_size,
                                                                    np.eye(3),
                                                                    np.zeros((4,1)))

    with open(args.out_file, 'w') as f:
        f.write('repError: {}\n'.format(repError))
        f.write('fx:{}\nfy:{}\ncx:{}\ncy:{}\n'.format(K[0,0], K[1,1], K[0,2], K[1,2]))
        f.write('D: {}\n'.format(D.flatten().tolist()))

    logging.info('Wrote output to {}'.format(args.out_file))

if __name__ == '__main__':
    main()