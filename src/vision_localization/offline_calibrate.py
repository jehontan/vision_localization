import cv2
import numpy as np
import argparse
import logging

marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

def main():
    parser = argparse.ArgumentParser(description='Camera calibration program.')
    parser.add_argument('in_file', type=str, help='NPZ file from remote node.')
    parser.add_argument('-o', dest='out_file', required=False, type=str, default='calibOutput.txt', help='Output file path. Default: "./calibOutput.txt"')
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

    data = np.load(args.in_file, allow_pickle=True)

    logging.info('Loaded {}'.format(args.in_file))

    board = cv2.aruco.CharucoBoard_create(int(data['board_cols']), int(data['board_rows']), float(data['chess_size']), float(data['marker_size']), marker_dict)

    repError, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(data['allCharucoCorners'].tolist(),
                                                                    data['allCharucoIds'].tolist(),
                                                                    board,
                                                                    data['img_size'],
                                                                    np.eye(3),
                                                                    np.zeros((5,1)))

    with open(args.out_file, 'w') as f:
                f.write('repError: {}\n'.format(repError))
                f.write('fx: {}\nfy: {}\ncx: {}\ncy: {}\n'.format(K[0,0], K[1,1], K[0,2], K[1,2]))
                f.write('D: {}\n'.format(D.flatten().tolist()))

    logging.info('Wrote calibration output to {}'.format(args.out_file))
    logging.info('Done.')

if __name__ == '__main__':
    main()