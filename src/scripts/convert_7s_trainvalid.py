import sys
sys.path.append("/home/siyanhu/Gits/mvsplat/src")
import root_file_io as fio

import time
import copy
import numpy as np
import subprocess
from collections import defaultdict
from colorama import Fore
from datetime import datetime

from typing import Literal, TypedDict
from jaxtyping import Float, Int, UInt8
import torch
from torch import Tensor
import json


TARGET_BYTES_PER_CHUNK = int(1e8)


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


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


def build_camera_info(intr, extr):
    downSample = 1.0
    scale_factor = 1.0 / 20

    intr[:2] *= 4
    intr[:2] = intr[:2] * downSample
    extr[:3, 3] *= scale_factor

    return intr, extr


def get_params(original_image_path, intrinsicss, extrinsicss, points3D={}, padding_factor=1):
    (filedir, file_name, fileext) = fio.get_filename_components(original_image_path)
    (nextfiledir, next_seq_name, next_fileext) = fio.get_filename_components(filedir)
    image_name = fio.sep.join([next_seq_name, file_name]) + '.' + fileext

    if image_name not in extrinsicss:
        print("[ERROR] Cannot find image in extrinsics", original_image_path, image_name)
        return
    
    near = 0.1
    far = 1000.0
    single_extrisic = extrinsicss[image_name]
    single_intrinsic = copy.deepcopy(intrinsicss[single_extrisic['camera_id']])

    qvec = extrinsicss[image_name]['qvec']
    R = quaternion_to_rotation_matrix(
        qvec[0], 
        qvec[1],
        qvec[2],
        qvec[3])
    
    t = extrinsicss[image_name]['tvec']

    near = 1.0
    far = 100.0
    if len(points3D) > 1:
        visible_points = [points3D[point3D_id]['xyz'] for point3D_id in points3D if extrinsicss[image_name]['image_id'] in points3D[point3D_id]['image_ids']]
        visible_points_array = np.array(visible_points)
        points_cam = np.dot(R, visible_points_array.T).T + t
        depths = points_cam[:, 2]
        min_depth = np.min(depths)
        max_depth = np.max(depths)
        near = max(min_depth / padding_factor, 0.1)
        far = max_depth * padding_factor

    intri, extri = build_camera_info(single_intrinsic['intrinsic'], single_extrisic['extrinsic'])
    return original_image_path, image_name, intri, extri, near, far


def load_raw(path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(image_paths) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    images_dict = {}
    for i in range(len(image_paths)):
        img_pth = image_paths[i]
        img_bin = load_raw(img_pth)
        images_dict[i] = img_bin
    return images_dict


def load_metadata(intrinsics, world2cams) -> Metadata:
    timestamps = []
    cameras = []
    url = ""

    for vid, intr in intrinsics.items():
        timestamps.append(int(vid))
        # normalized the intr
        fx = intr[0, 0]
        fy = intr[1, 1]
        cx = intr[0, 2]
        cy = intr[1, 2]
        w = 2.0 * cx
        h = 2.0 * cy
        saved_fx = fx / w
        saved_fy = fy / h
        saved_cx = 0.5
        saved_cy = 0.5
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]

        # print(vid)

        w2c = world2cams[vid]
        camera.extend(w2c[:3].flatten().tolist())
        cameras.append(np.array(camera))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


def get_size(path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


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
    this_time = fio.get_current_timestamp("%Y_%m_%d")
    testing_count = 0

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

    branch = 'test_full_byorder_59'
    test_intri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'cameras.txt')
    test_intri_dict = get_colmap_intrinsic(test_intri_pth)
    test_extri_pth = fio.createPath(fio.sep,[scene_data_dir, branch, 'sparse', '0'], 'images.txt')
    test_extri_dict = get_colmap_extrinsic(test_extri_pth)

    output_dir = fio.createPath(fio.sep, [fio.getParentDir(), "datasets", data_tag, scene_tag])
    fio.ensure_dir(output_dir)

    paths, relation = process_scene(scene_data_dir, scene_pair_path)

    for target_img_label in relation.keys():

        test_img_path = paths[target_img_label]
        update_img_pth, vid_name, intrins, extrins, near, far = get_params(test_img_path, test_intri_dict, test_extri_dict)
        
        img_bin_test = load_raw(test_img_path)
        num_bytes_test = get_size(test_img_path) // 7

        example_test = load_metadata({"0":intrins}, {"0":extrins})
        example_test["images"] = [
            img_bin_test
        ]

        example_test["key"] = target_img_label
        save_path_torch_test = fio.createPath(fio.sep, [output_dir, this_time, target_img_label.replace('.png', ''), 'test'], '000000.torch')
        (sptt_dir, sptt_name, sptt_ext) = fio.get_filename_components(save_path_torch_test)
        fio.ensure_dir(sptt_dir)
        print(save_path_torch_test)
        torch.save([example_test], save_path_torch_test)

        save_path_index_test = fio.createPath(fio.sep, [output_dir, this_time, target_img_label.replace('.png', ''), 'test'], 'index.json')
        save_json_test = {}
        save_json_test[target_img_label] = '000000.torch'
        with open(save_path_index_test, 'w') as file0:
            json.dump(save_json_test, file0, indent=4)

        vid_dict_train, intrinsics_train, world2cams_train, cam2worlds_train, near_fars_train = {}, {}, {}, {}, {}
        images_dict_train = {}
        training_img_labels = relation[target_img_label]

        output_dir_sub0 = fio.createPath(fio.sep, [output_dir, this_time, target_img_label.replace('.png', ''), 'train', 'images'])
        fio.ensure_dir(output_dir_sub0)

        for ti_vid in range(len(training_img_labels)):
            ti_label = training_img_labels[ti_vid]
            ti_image_path = paths[ti_label]

            if fio.file_exist(ti_image_path) == False:
                continue
            
            move_to_path = fio.createPath(fio.sep, [output_dir_sub0], ti_label)
            (move_to_pathdir, move_to_pathname, move_to_pathext) = fio.get_filename_components(move_to_path)
            fio.ensure_dir(move_to_pathdir)
            fio.copy_file(ti_image_path, move_to_path)

            update_img_pth, vid_name, intrins, extrins, near, far = get_params(ti_image_path, train_intri_dict, train_extri_dict, train_pnt3d_dict)
            
            img_bin = load_raw(ti_image_path)
            images_dict_train[ti_vid] = img_bin

            vid_dict_train[ti_vid] = vid_name
            intrinsics_train[ti_vid] = intrins
            world2cams_train[ti_vid] = extrins
            cam2worlds_train[ti_vid] = np.linalg.inv(extrins)
            near_fars_train[ti_vid] = [near, far]
        
        image_dir = fio.getParentDir(output_dir_sub0)
        num_bytes = get_size(image_dir) // 7

        example = load_metadata(intrinsics_train, world2cams_train)
        example["images"] = [
            images_dict_train[timestamp.item()] for timestamp in example["timestamps"]
        ]

        assert len(images_dict_train) == len(example["timestamps"])
        example["key"] = target_img_label

        save_path_torch = fio.createPath(fio.sep, [output_dir, this_time, target_img_label.replace('.png', ''), 'train'], '000000.torch')
        print(save_path_torch)
        torch.save([example], save_path_torch)
        fio.delete_folder(output_dir_sub0)
        
        save_path_index = fio.createPath(fio.sep, [output_dir, this_time, target_img_label.replace('.png', ''), 'train'], 'index.json')
        save_json = {}
        save_json[target_img_label] = '000000.torch'
        with open(save_path_index, 'w') as file:
            json.dump(save_json, file, indent=4)

        testing_count += 1

        if testing_count > 10:
            break