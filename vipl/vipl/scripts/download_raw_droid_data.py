import json
import os
import random
import glob
import h5py
import cv2
import shutil
import subprocess
from PIL import Image
import io
import numpy as np
import tqdm

data_manifest_location = "~/VISTA/manifest.json"
temp_download_location = "~/droid_data_raw_temp"
parsed_data_location = "~/droid_data_multiview/"
parsed_data_manifest_location = parsed_data_location + "manifest.json"

MAX_ITEMS_PER_FILE = 1000
SAMPLES_PER_TRAJECTORY = 5

def download_file_from_bucket(file_name, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    command = f"gsutil -m cp {file_name} {local_dir}"
    # print("Running command: ")
    # print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired as e:
        print(f"Command {command} timed out.")
        return False
    output = process.stderr.read().decode('utf-8')
    print(output)
    if "CommandException" in output:
        return False
    return True

def get_cam_serials_from_metadata(metadata):
    return metadata["ext1_cam_serial"], metadata["ext2_cam_serial"]


def download_trajectory(gsutil_path):
    # we'll download into a temporary directory which is the last part of the gsutil path
    tmpdir = os.path.join(temp_download_location, gsutil_path.split('/')[-2])
    os.makedirs(tmpdir, exist_ok=True)
    # first download the metadata
    metadata_gsutil_path = os.path.join(gsutil_path, "metadata*")
    metadata_dl_success = download_file_from_bucket(metadata_gsutil_path, tmpdir)
    if not metadata_dl_success:
        return tmpdir, None
    metadata_file = glob.glob(os.path.join(tmpdir, "metadata*"))[0]
    if not os.path.exists(metadata_file):
        print(f"Failed to download metadata file for {gsutil_path}")
        return tmpdir, None
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    cam_serials = get_cam_serials_from_metadata(metadata)
    # then download the videos
    for cam_serial in cam_serials:
        video_gsutil_path = os.path.join(gsutil_path, f"recordings/MP4/{cam_serial}-stereo.mp4")
        download_file_from_bucket(video_gsutil_path, tmpdir)
    # also download trajectory.h5
    download_file_from_bucket(os.path.join(gsutil_path, "trajectory.h5"), tmpdir)
    return tmpdir, metadata


def validate_trajectory(trajectory_path, metadata):
    # make sure that there are videos for each serial number in the folder trajectory_path
    cam_serials = get_cam_serials_from_metadata(metadata)
    for cam_serial in cam_serials:
        video_path = os.path.join(trajectory_path, f"{cam_serial}-stereo.mp4")
        if not os.path.exists(video_path):
            print(f"Failed to find video for {cam_serial} in {trajectory_path}")
            return False
    if not os.path.exists(os.path.join(trajectory_path, "trajectory.h5")):
        print(f"Failed to find trajectory.h5 in {trajectory_path}")
        return False
    return True


def get_frames_from_video(mp4_path, frame_idxs):
    if not isinstance(frame_idxs, list):
        frame_idxs = [frame_idxs]
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);
    frames = []
    for frame_idx in frame_idxs:
        # import time
        # t0 = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # t1 = time.perf_counter()
        # print("time to seek", t1-t0)
        ret, frame = cap.read()
        frames.append(frame)
    cap.release()
    return frames


def split_frame(frame):
    # split a frame in half into two images with each half the width of the original
    height, width, _ = frame.shape
    half_width = width // 2
    left_half = frame[:, :half_width]
    right_half = frame[:, half_width:]
    return left_half, right_half


def cv2_image_to_jpg_bytes(image):
    # Convert the frame from cv2 to PIL image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # BytesIO is a file-like buffer stored in memory
    img_byte_arr = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(img_byte_arr, format='jpeg')
    # Turn the BytesIO object back into a bytes object
    img_byte_arr = img_byte_arr.getvalue()
    img_np_array = np.asarray(img_byte_arr)
    return img_np_array


def process_trajectory(path_to_save_to, idx, trajectory_path, metadata, N=5):
    # process the trajectory
    # 1. extract the frames from the videos
    # 2. extract the metadata from the metadata file
    # 3. write to the hdf5 file.
    # create a new group in the hdf5 file
    # first load the trajectory.h5 file
    trajectory = h5py.File(os.path.join(trajectory_path, "trajectory.h5"), "r")
    cam_serials = get_cam_serials_from_metadata(metadata)
    # open the output h5 file
    hdf5_to_save_to = h5py.File(path_to_save_to, "a")
    # print("Saving to ", path_to_save_to)
    # get the length of each video, which is the shape of trajectory[/observation/timestamp/cameras/16291792_estimated_capture]
    video_length = trajectory[f"/observation/timestamp/cameras/{cam_serials[0]}_estimated_capture"].shape[0] - 1
    if video_length < N:
        return False
    random_frame_indices = sorted(random.sample(range(video_length), N))
    # compute indices to read for each camera view
    idxs_to_read = {k: [] for k in cam_serials}
    idxs_to_read[cam_serials[0]] = random_frame_indices
    # the ones for the first serial number are randomly selected
    # the rest are computed based on closeness in recording time to the ones for the first serial number
    timestamps = [trajectory[f"/observation/timestamp/cameras/{cam_serials[0]}_estimated_capture"][j] for j in random_frame_indices]
    for cam_serial in cam_serials[1:]:
        for sample in range(N):
            # the idx to read is the one with the closest timestamp to timestamp
            camera_times = trajectory[f"/observation/timestamp/cameras/{cam_serial}_estimated_capture"]
            closest_idx = np.argmin(np.abs(camera_times - timestamps[sample]))
            idxs_to_read[cam_serial].append(closest_idx)

    camera_frames = dict()
    import time
    start_time = time.time()
    for cam_serial in cam_serials:
        camera_frames[cam_serial] = get_frames_from_video(os.path.join(trajectory_path, f"{cam_serial}-stereo.mp4"), idxs_to_read[cam_serial])
    end_time = time.time()
    print(f"Processed {len(cam_serials)} cameras in {end_time - start_time} seconds.")
    # camera frames is a dict where keys are camera serial numbers and values are a list of frames corresponding to each sample, correspondingly.

    for sample in range(N):
        # save the group to the hdf5 output file for this sample
        group = hdf5_to_save_to.create_group(f"sample_{idx + sample}")
        # print("Group name", idx + sample)
        # save the metadata dict as part of group attrs
        group.attrs.update(metadata)
        cam_images, cam_extrinsics, read_indices = dict(), dict(), dict()
        for cam_serial in cam_serials:
            frame = camera_frames[cam_serial][sample]
            # this is actually going to be two frames side by side
            left, right = split_frame(frame) 
            cam_images.update({
                f'{cam_serial}_left': left,
                f'{cam_serial}_right': right,
            })
            cam_extrinsics[f'{cam_serial}_left'] = trajectory[f"/observation/camera_extrinsics/{cam_serial}_left"][idxs_to_read[cam_serial][sample]]
            cam_extrinsics[f'{cam_serial}_right'] = trajectory[f"/observation/camera_extrinsics/{cam_serial}_right"][idxs_to_read[cam_serial][sample]]
            read_indices[f'{cam_serial}_left'] = idxs_to_read[cam_serial][sample]
            read_indices[f'{cam_serial}_right'] = idxs_to_read[cam_serial][sample]
        # convert cam_images to jpg bytes
        cam_images_jpg = {k: cv2_image_to_jpg_bytes(v) for k, v in cam_images.items()}
        # now let's write everything to the hdf5 file
        # write a hdf5 dataset into the group for each image and each camera extrinsic
        for k, v in cam_images_jpg.items():
            group.create_dataset(k, data=v)
        for k, v in cam_extrinsics.items():
            group.create_dataset(f"{k}_extrinsics", data=v)
            group.create_dataset(f"{k}_idx", data=read_indices[k])
    hdf5_to_save_to.close()
    return True


def download_and_process_data(gsutil_path, file_to_save_to, idx, N):
    success = False
    print(f"Processing {gsutil_path}:")
    # download data
    print("\t Downloading files...")
    trajectory_path, metadata = download_trajectory(gsutil_path)
    if metadata is not None:
        print("\t \t OK")
    else:
        print("\t \t Download failed")
        shutil.rmtree(trajectory_path)
        return False 
    print("\t Validating...")
    trajectory_valid = validate_trajectory(trajectory_path, metadata)
    # process data
    if trajectory_valid:
        print("\t \t OK")
        print("\t Processing trajectory...")
        success = process_trajectory(file_to_save_to, idx, trajectory_path, metadata, N=N)
        if success:
            print("\t \t OK")
        else:
            print("\t \t Processing failed.")
    else:
        print("\t \t Trajectory invalid.")
    # delete the trajectory path folder
    shutil.rmtree(trajectory_path)
    return success

def convert_data_manifest_path_to_gs_bucket(path):
    # The path loks like this: 
    # /mnt/fsx/surajnair/datasets/r2d2_full_raw/GuptaLab/success/2023-07-08/Mon_Jul_10_09:38:30_2023/trajectory_im128.h5
    # this should be converted to something like
    # gs://gresearch/robotics/droid_raw/1.0.1/GuptaLab/success/2023-07-08/Mon_Jul_10_09:38:30_2023/
    path_parts = path.split('/')
    gs_bucket_path = "gs://gresearch/robotics/droid_raw/1.0.1/" + '/'.join(path_parts[6:-1]) + '/'
    return gs_bucket_path

def clean_data_manifest(data_manifest):
    data_list = map(convert_data_manifest_path_to_gs_bucket, [traj["path"] for traj in data_manifest])
    return list(data_list)

def list_to_hashmap(data_list):
    return {data: True for data in data_list}

def hashmap_to_list(map):
    return [data for data in map if map[data]]

def next_file_index_to_create(data_folder):
    i = 0
    while os.path.exists(os.path.join(data_folder, f"data_{i}.h5")):
        i += 1
    return i

def download_raw_droid_data():
    data_manifest = json.load(open(data_manifest_location, "r"))
    data_manifest = clean_data_manifest(data_manifest)
    # shuffle data manifest
    random.shuffle(data_manifest)

    # if parsed data manifest doesn't exist, create it
    if not os.path.exists(parsed_data_manifest_location):
        os.makedirs(os.path.dirname(parsed_data_manifest_location), exist_ok=True)
        with open(parsed_data_manifest_location, "w") as f:
            json.dump({}, f)

    # load current parsed data manifest
    parsed_data_manifest = list(json.load(open(parsed_data_manifest_location, "r")))
    # create a hashmap for which data has already been parsed
    parsed_data_manifest_map = list_to_hashmap(parsed_data_manifest)

    num_processed = 0 
    file_idx_to_create = next_file_index_to_create(parsed_data_location)
    file_to_save_to = os.path.join(parsed_data_location, f"data_{file_idx_to_create}.h5")
    current_sample_index = 0

    # for each data in data_manifest, if it's not in parsed_data_manifest, download it, process it, and add it to parsed_data_manifest
    for gsutil_path in data_manifest:
        if gsutil_path not in parsed_data_manifest_map:
            processed_sucessfully = download_and_process_data(gsutil_path, file_to_save_to, current_sample_index, N=SAMPLES_PER_TRAJECTORY)
            # add data to parsed_data_manifest
            parsed_data_manifest_map[gsutil_path] = True
            num_processed += 1
            if num_processed % 5 == 0:
                # update parsed_data_manifest
                print(f"Processed {num_processed} files, updating manifest...")
                parsed_data_manifest = hashmap_to_list(parsed_data_manifest_map)
                with open(parsed_data_manifest_location, "w") as f:
                    json.dump(parsed_data_manifest, f)
    
            if processed_sucessfully:
                current_sample_index += SAMPLES_PER_TRAJECTORY
                if current_sample_index >= MAX_ITEMS_PER_FILE:
                    file_idx_to_create += 1
                    file_to_save_to = os.path.join(parsed_data_location, f"data_{file_idx_to_create}.h5")
                    current_sample_index = 0

if __name__ == "__main__" :
    download_raw_droid_data()
