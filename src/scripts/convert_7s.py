import sys
sys.path.append("/home/siyanhu/Gits/mvsplat/src")
import root_file_io as fio

import time
import numpy as np
from collections import defaultdict
from colorama import Fore
from datetime import datetime


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def get_current_timestamp(format_str=''):
    if format_str == '':
        # Return timestamp as integer in milliseconds
        return int(time.time() * 1000)
    else:
        # Return formatted timestamp string
        return datetime.now().strftime(format_str)


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert a quaternion to a rotation matrix."""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def colmap_intrinsics_to_camera_matrix(camera_model, params):
    """
    Convert COLMAP intrinsics format to a 3x3 camera matrix.
    
    Args:
    camera_model: String representing the COLMAP camera model
    params: List of parameters specific to the camera model
    
    Returns:
    3x3 camera matrix
    """
    if camera_model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
    
    elif camera_model == "PINHOLE":
        fx, fy, cx, cy = params
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    elif camera_model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        # Note: The radial distortion parameter k is not used in the camera matrix
        return np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
    
    else:
        raise ValueError(f"Unsupported camera model: {camera_model}")
    

def colmap_extrinsics_to_transformation_matrix(qw, qx, qy, qz, tx, ty, tz):
    """
    Convert COLMAP extrinsics format to a 4x4 transformation matrix.
    
    Args:
    qw, qx, qy, qz: Quaternion components
    tx, ty, tz: Translation components
    
    Returns:
    4x4 transformation matrix
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    
    # Create translation vector
    t = np.array([[tx], [ty], [tz]])
    
    # Construct 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    return T


def get_colmap_extrinsic(extri_file):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(extri_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_name] = {
                    "image_id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "extrinsic": colmap_extrinsics_to_transformation_matrix(
                        qvec[0], 
                        qvec[1],
                        qvec[2],
                        qvec[3],
                        tvec[0],
                        tvec[1],
                        tvec[2]
                        ),
                    "camera_id": camera_id,
                    "image_name": image_name,
                    "xys": xys,
                    "point3d_ids": point3D_ids
                }
    return images


def get_colmap_intrinsic(intri_file):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(intri_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])

                if camera_id in cameras:
                    # print(len(cameras))
                    continue

                model = elems[1]
                # assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = {
                    "camera_id": camera_id,
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params,
                    "intrinsic": colmap_intrinsics_to_camera_matrix(model, params)
                }
    return cameras


def get_colmap_points3d(pnt3d_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    image_points = defaultdict(list)
    with open(pnt3d_file, 'r') as f:
        for line in f:
            if line[0] != '#':
                data = line.split()
                point3D_id = int(data[0])
                x, y, z = map(float, data[1:4])
                r, g, b = map(int, data[4:7])
                error = float(data[7])
                image_ids = list(map(int, data[8::2]))
                point2D_idxs = list(map(int, data[9::2]))
                points3D[point3D_id] = {'xyz': [x, y, z], 'rgb': [r, g, b], 'error': error, 'image_ids': image_ids, 'point2D_idxs': point2D_idxs}
    return points3D


def parse_pairs_file(filename, data_dir):
    pairs_path_dict = {}
    pairs_label_dict = {}
    
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into key and value
            key, value = line.strip().split()
            key_path = fio.createPath(fio.sep, [data_dir, 'test_full_byorder_59', 'images'], key)
            value_path = fio.createPath(fio.sep, [data_dir, 'train_full_byorder_85', 'images'], value)
            
            if (fio.file_exist(key_path) == False) or (fio.file_exist(value_path) == False):
                continue
            if key_path not in pairs_path_dict:
                pairs_path_dict[key] = key_path
            if value_path not in pairs_path_dict:
                pairs_path_dict[value] = value_path

            if key not in pairs_label_dict:
                pairs_label_dict[key] = []
            pairs_label_dict[key].append(value)
    return pairs_path_dict, pairs_label_dict


def process_scene(data_dir, pair_path):
    (datadir_dir, datadir_name, datadir_ext) = fio.get_filename_components(data_dir)
    print(cyan("Processing conversion for scene: {}".format(datadir_name)))
    path_dict, relation_dict = parse_pairs_file(pair_path, data_dir)
    return path_dict, relation_dict


# def find_parameters(seq_name, )

if __name__ == '__main__':
    data_tag = '7s'
    scene_tag = 'scene_stairs'

    scene_data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'datasets_raw', data_tag, scene_tag])
    scene_pair_path = fio.createPath(fio.sep, [fio.getParentDir(), 'datasets_pairs', data_tag, scene_tag], 'pairs-query-netvlad10.txt')

    if (fio.file_exist(scene_data_dir) == False) or (fio.file_exist(scene_pair_path) == False):
        print(cyan("[ERROR] No data detected: {}, {}". format(data_tag, scene_tag)))
        exit()

    branch = 'train_full_byorder_85'
    train_intri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'cameras.txt')
    train_intri_dict = get_colmap_intrinsic(train_intri_pth)
    train_extri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'images.txt')
    train_extri_dict = get_colmap_extrinsic(train_extri_pth)
    train_pnt3d_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'points3D.txt')
    train_pnt3d_dict = get_colmap_points3d(train_pnt3d_pth)

    test_intri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'cameras.txt')
    test_intri_dict = get_colmap_intrinsic(test_intri_pth)
    test_extri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'images.txt')
    test_extri_dict = get_colmap_extrinsic(test_extri_pth)

    # output_dir = fio.createPath(fio.sep, [fio.getParentDir(), scene_data_dir, data_tag + '_pair_' + scene_tag])
    # fio.ensure_dir(output_dir)

    paths, relation = process_scene(scene_data_dir, scene_pair_path)

    for target_img_label in relation.keys():
        training_img_labels = relation[target_img_label]
        print(training_img_labels)

        