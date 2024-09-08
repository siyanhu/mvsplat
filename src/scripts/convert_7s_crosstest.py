import sys
sys.path.append("/home/siyanhu/Gits/mvsplat/src")
import root_file_io as fio
import tqdm
import copy
import re
import subprocess
import numpy as np
from collections import defaultdict
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


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


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


def calculate_near_far(images, points3D, padding_factor=1.1):
    min_depth = float('inf')
    max_depth = float('-inf')
    
    for image_id, image_data in images.items():
        R = quaternion_to_rotation_matrix(image_data['qvec'])
        t = np.array(image_data['tvec'])
        
        # Get points visible in this image
        visible_points = [points3D[point3D_id]['xyz'] for point3D_id in points3D if image_id in points3D[point3D_id]['image_ids']]
        
        if not visible_points:
            continue
        
        points_cam = np.dot(R, np.array(visible_points).T).T + t
        depths = points_cam[:, 2]
        
        min_depth = min(min_depth, np.min(depths))
        max_depth = max(max_depth, np.max(depths))
    
    near = max(min_depth / padding_factor, 0.1)
    far = max_depth * padding_factor
    
    return near, far


def build_camera_info(intr, extr):
    downSample = 1.0
    scale_factor = 1.0 / 20

    intr[:2] *= 4
    intr[:2] = intr[:2] * downSample
    extr[:3, 3] *= scale_factor

    return intr, extr


def get_original_params(original_image_path, intrinsicss, extrinsicss, points3D, padding_factor=1):
    # (filedir, filename, fileext) = fio.get_filename_components(mvsp_image_path)
    # grand_parent_folder = fio.getGrandParentDir(mvsp_image_path)

    # original_image_path = fio.createPath(fio.sep, [grand_parent_folder, 'images', 'seq-01'], '.'.join([filename, fileext]))
    # if (fio.file_exist(original_image_path)) == False:
    #     original_image_path = fio.createPath(fio.sep, [grand_parent_folder, 'images', 'seq-02'], '.'.join([filename, fileext]))
    #     if (fio.file_exist(original_image_path)) == False:
    #         print("[ERROR] File doesn't exist", original_image_path)
    #         return
    
    # original_parent_dir = fio.getParentDir(original_image_path)
    # (filedir, filename, fileext) = fio.get_filename_components(original_parent_dir)
    # original_seq_name = filename
    # (filedir, filename, fileext) = fio.get_filename_components(original_image_path)
    # original_image_name = filename
    # original_image_ext = fileext
    # image_name = fio.sep.join([original_seq_name, original_image_name]) + '.' + original_image_ext

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

    visible_points = [points3D[point3D_id]['xyz'] for point3D_id in points3D if extrinsicss[image_name]['image_id'] in points3D[point3D_id]['image_ids']]
    visible_points_array = np.array(visible_points)
    qvec = extrinsicss[image_name]['qvec']
    R = quaternion_to_rotation_matrix(
        qvec[0], 
        qvec[1],
        qvec[2],
        qvec[3])
    
    t = extrinsicss[image_name]['tvec']
    points_cam = np.dot(R, visible_points_array.T).T + t
    depths = points_cam[:, 2]
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    near = max(min_depth / padding_factor, 0.1)
    far = max_depth * padding_factor

    intri, extri = build_camera_info(single_intrinsic['intrinsic'], single_extrisic['extrinsic'])
    return original_image_path, image_name, intri, extri, near, far


def get_size(path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))

# from PIL import Image
# from torchvision import transforms
# from torch import nn
# import torch.nn.functional as F

# def load_raw(path) -> UInt8[Tensor, " length"]:
#     # Load the raw image data
#     raw_data = np.memmap(path, dtype="uint8", mode="r")
#     tensor = torch.tensor(raw_data, dtype=torch.uint8)
    
#     # Reshape to the original dimensions (480, 640)
#     tensor = tensor.view(480, 640)
    
#     # Add a batch dimension and a channel dimension
#     tensor = tensor.unsqueeze(0).unsqueeze(0)
    
#     # Resize to (360, 640)
#     resized_tensor = nn.functional.interpolate(tensor, size=(360, 640), mode='bilinear', align_corners=False)
    
#     # Remove the batch and channel dimensions
#     resized_tensor = resized_tensor.squeeze(0).squeeze(0)
    
#     return resized_tensor

#     return resized_image

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


def sort_file_names(file_list):
    def extract_index(file_path):
        (filedar, file_name, fileext) = fio.get_filename_components(file_path)
        match = re.search(r'frame-(\d+)', file_name)
        return int(match.group(1)) if match else 0

    return sorted(file_list, key=extract_index)


def sort_and_sample_file_names(file_list, sample_rate=200):
    def extract_index(file_name):
        match = re.search(r'frame-(\d+)', file_name)
        return int(match.group(1)) if match else 0

    sorted_files = sorted(file_list, key=extract_index)
    
    N = len(sorted_files)
    num_samples = max(2, int(N / sample_rate))
    
    if num_samples >= N:
        return sorted_files
    
    step = N / num_samples
    sampled_indices = [int(i * step) for i in range(num_samples)]
    
    return [sorted_files[i] for i in sampled_indices]


def traverse_image_folder(image_dir):
    seq_dir_paths = fio.traverse_dir(image_dir, full_path=True, towards_sub=False)
    seq_dir_paths = fio.filter_folder(seq_dir_paths, filter_out=False, filter_text='seq')

    image_seq_dict = {}
    image_sample_dict = {}
    for seq_dir in seq_dir_paths:
        (seqdirdir, seqdirname, seqdirext) = fio.get_filename_components(seq_dir)
        image_paths = fio.traverse_dir(seq_dir, full_path=True, towards_sub=False)
        image_paths = fio.filter_ext(image_paths, filter_out_target=False, ext_set=fio.img_ext_set)
        image_paths = sort_file_names(image_paths)

        # image_sample_paths = sort_and_sample_file_names(image_paths)

        image_target_paths = sort_and_sample_file_names(image_paths, sample_rate=int(len(image_paths) / 50))
        image_base_paths = [image_target_paths[0], image_target_paths[-1]]

        image_seq_dict[seqdirname] = image_target_paths
        image_sample_dict[seqdirname] = image_base_paths
    return image_seq_dict, image_sample_dict


if __name__ == '__main__':
    data_tag = '7s'
    data_dir = fio.createPath(fio.sep, ['datasets_raw', data_tag])
    scene_dirs = fio.traverse_dir(data_dir, full_path=True, towards_sub=False)
    scene_dirs = fio.filter_folder(scene_dirs, filter_out=False, filter_text='scene_')

    branch = 'train_full_byorder_85'
    this_time = fio.get_current_timestamp("%Y_%m_%d")
    
    for scene_dir_pth in scene_dirs:
        (scenedir, scenename, sceneext) = fio.get_filename_components(scene_dir_pth)
        
        refresh_folder = fio.createPath(fio.sep, ['datasets_crossvalid', scenename, this_time, 'test'])
        if fio.file_exist(refresh_folder):
            fio.delete_folder(refresh_folder)
        fio.ensure_dir(refresh_folder)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                # f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
                 f"Saving chunk {chunk_key} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = fio.createPath(fio.sep, ['datasets_crossvalid', scenename, this_time, 'test'])
            torch.save(chunk, dir + fio.sep + "{}.torch".format(chunk_key))
            chunk_size = 0
            chunk_index += 1
            chunk = []

        (scenedir, scenename, sceneext) = fio.get_filename_components(scene_dir_pth)
        image_dir = fio.createPath(fio.sep,[scene_dir_pth, branch, 'images'])
        image_seq_dict, image_sample_dict = traverse_image_folder(image_dir)
        
        intri_pth = fio.createPath(fio.sep,[scene_dir_pth, branch, 'sparse', '0'], 'cameras.txt')
        intri_dict = get_colmap_intrinsic(intri_pth)
        extri_pth = fio.createPath(fio.sep,[scene_dir_pth, branch, 'sparse', '0'], 'images.txt')
        extri_dict = get_colmap_extrinsic(extri_pth)
        pnt3d_pth = fio.createPath(fio.sep,[scene_dir_pth, branch, 'sparse', '0'], 'points3D.txt')
        pnt3d_dict = get_colmap_points3d(pnt3d_pth)

        vid_dict, intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}, {}
        images_dict = {}
        base_vids_dict = {}
        target_vids_dict = {}

        for key, img_seq in image_seq_dict.items():
            image_dir = ''

            base_vids = []
            target_vids = []

            if key not in image_seq_dict:
                continue
            sample_seq = image_sample_dict[key]

            for vid in range(len(img_seq)):
                img_pth = img_seq[vid]
                if len(image_dir) < 1:
                    image_dir = fio.getParentDir(img_pth)

                update_img_pth, vid_name, intrins, extrins, near, far = get_original_params(img_pth, intri_dict, extri_dict, pnt3d_dict)
                global_vid = vid

                if img_pth in sample_seq:
                    base_vids.append(global_vid)
                else:
                    target_vids.append(global_vid)

                img_bin = load_raw(img_pth)
                images_dict[global_vid] = img_bin

                vid_dict[global_vid] = vid_name
                intrinsics[global_vid] = intrins
                world2cams[global_vid] = extrins
                cam2worlds[global_vid] = np.linalg.inv(extrins)
                near_fars[global_vid] = [near, far]

            base_vids_dict[key] = base_vids
            target_vids_dict[key] = target_vids

            image_dir = fio.getParentDir(img_pth)
            num_bytes = get_size(image_dir) // 7

            example = load_metadata(intrinsics, world2cams)
            example["images"] = [
                images_dict[timestamp.item()] for timestamp in example["timestamps"]
            ]

            assert len(images_dict) == len(example["timestamps"])
            example["key"] = key
            
            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()

        print("Generate key:torch index...")
        index = {}
        save_dir = fio.createPath(fio.sep, ['datasets_crossvalid', scenename, this_time, 'test'])
        fio.ensure_dir(save_dir)
        save_path = fio.createPath(fio.sep, [save_dir], 'index.json')

        chunk_paths = fio.traverse_dir(save_dir, full_path=True, towards_sub=False)
        chunk_paths = fio.filter_ext(chunk_paths, filter_out_target=False, ext_set=['torch'])

        for chnk_pth in chunk_paths:
            (chnkfile, chnkname, chnkext) = fio.get_filename_components(chnk_pth)
            chunk = torch.load(chnk_pth)
            for example in chunk:
                index[example["key"]] = chnkname + '.' + chnkext

            with open(save_path, 'w') as f:
                json.dump(index, f)

        save_dir1 = fio.createPath(fio.sep, ['datasets_crossvalid', scenename, this_time, 'test'])
        fio.ensure_dir(save_dir1)
        save_path1 = fio.createPath(fio.sep, [save_dir1], '_'.join(["evaluation_index", '7s', scenename]) + '.json')

        if fio.file_exist(save_path1):
            fio.delete_file(save_path1)

        # {"seq-02": {"context": [1, 34, 60], "target": [2, 3, 23,45, 48]}}
        # sample_seq = 
        vidkeys = list(base_vids_dict.keys())
        asset_dict = {}
        for vdk in vidkeys:
            base_vid_list = base_vids_dict[vdk]
            target_vid_list = target_vids_dict[vdk]

            for vi in range(len(base_vid_list) - 1):
                id0 = base_vid_list[vi]
                id1 = base_vid_list[vi + 1]

                possible_targets = list(range(id0 + 1, id1))
                confirmed_targets = possible_targets[:] = [x for x in possible_targets if x in target_vid_list]
                confirmed_targets_slice = confirmed_targets[: 5]

                asset_dict[vdk] = {
                    "context": [id0, id1],
                    "target": confirmed_targets
                }

        with open(save_path1, 'w') as file:
            json.dump(asset_dict, file, indent=4)