import datetime
import glob
import os
import pickle
import cv2

import h5py
import hydra
import numpy as np
from tqdm import trange, tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R
import robomimic
from robomimic.utils.lang_utils import get_lang_emb

def quat_to_euler(quat, degrees=False):
        euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
        return euler

def add_angles(delta, source, degrees=False):
        delta_rot = R.from_euler("xyz", delta, degrees=degrees)
        source_rot = R.from_euler("xyz", source, degrees=degrees)
        new_rot = delta_rot * source_rot
        return new_rot.as_euler("xyz", degrees=degrees)

def shortest_angle(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def normalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 2 * (arr - min_val) / (max_val - min_val) - 1


def unnormalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 0.5 * (arr + 1) * (max_val - min_val) + min_val

#just for checking the right keys
def print_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as file:
        def recurse(group, indent=0):
            for key in group.keys():
                print('  ' * indent + key)
                if isinstance(group[key], h5py.Group):
                    recurse(group[key], indent + 1)
                elif isinstance(group[key], h5py.Dataset):
                    print('  ' * (indent + 1) + "<Dataset>")

        recurse(file)


# Since LIBERO doesn't have language instructions inside the hdf5 file.
def extract_instruction_from_filename(filename):
    instruction = os.path.basename(filename).replace('.hdf5', '')
    replace_patterns = [
        'KITCHEN_SCENE', 'LIVING_ROOM_SCENE', 'STUDY_SCENE', 'demo',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'
    ]
    for pattern in replace_patterns:
        instruction = instruction.replace(pattern, '')
    instruction = instruction.replace('_', ' ')
    instruction = instruction.strip()
    instruction = ' '.join(instruction.split())
    return instruction


@hydra.main(
    config_path="../../configs/", config_name="convert_demos_real", version_base="1.1"
)
def run_experiment(cfg):

    # REMOVE ME !!!!
    # print_hdf5_structure('/home/raymond/LIBERO/datasets/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5')
    cfg.data_dir = "/home/raymond/LIBERO/datasets/libero_90"
    #concat all the hdf5 files together
    cfg.input_datasets = glob.glob(os.path.join(cfg.data_dir, '*.hdf5'))
    cfg.output_dataset = "all_libero_90"
    # print("Found files:", cfg.input_datasets)
    # REMOVE ME !!!!

    # create dataset paths
    dataset_paths = [
        os.path.join(cfg.data_dir, dataset_name) for dataset_name in cfg.input_datasets
    ]
    # print(dataset_paths)

    print("Processing data ...")

    hdf5_path = os.path.join(cfg.data_dir, cfg.output_dataset)
    os.makedirs(hdf5_path, exist_ok=True)
    f = h5py.File(os.path.join(hdf5_path, "demos.hdf5"), "w")

    # create data group
    grp = f.create_group("data")
    grp_mask = f.create_group("mask")

    episodes = 0

    demo_keys = {}

    for split in cfg.splits:

        demo_keys[split] = []

        for dataset_path in dataset_paths:

            print(f"Loading {dataset_path} {split} ...")

            # gather filenames
            # file_name = os.path.join(dataset_path, split, "demo.hdf5")
            file_name = os.path.join(dataset_path)
            data = h5py.File(file_name,'r')
            keys = data["data"].keys()
            for i, curr_key in tqdm(enumerate(keys), total=len(keys)):
                # load data
                # data = np.load(file_names[i], allow_pickle=True)
                # data = h5py.File(file_names[i],'r')
                # demo_key = "demo_0"

                world_offset_pos = np.array([0.2045, 0., 0.])
                ee_offset_euler = np.array([0., 0., -np.pi / 4])

                #changed all keys to "correct" keys present in libero

                ee_pos = np.array(data["data"][curr_key]["obs"]["ee_pos"]).copy()
                # add pos offset
                ee_pos = ee_pos + world_offset_pos
                # print(data["data"][curr_key]["obs"]["ee_states"].shape)
                # print(data["data"][curr_key]["obs"]["ee_pos"].shape)
                # print(data["data"][curr_key]["obs"]["ee_ori"].shape)
                # ee_quat = np.array(data["data"][curr_key]["obs"]["ee_states"]).copy()
                # convert quat to euler
                # ee_euler = quat_to_euler(ee_quat)
                ee_euler = np.array(data["data"][curr_key]["obs"]["ee_ori"]).copy()
                # add angle offset
                ee_euler = add_angles(ee_offset_euler, ee_euler)

                # qpos = np.array(data["data"][curr_key]["obs"]["joint_pos"]).copy()
                qpos = np.array(data["data"][curr_key]["obs"]["joint_states"]).copy()
                
                actions = np.array(data["data"][curr_key]["actions"]).copy()

                #gripper = np.array(data["data"][curr_key]["obs"]["gripper_qpos"]).copy()
                gripper = np.array(data["data"][curr_key]["obs"]["gripper_states"]).copy()
                
                # imgs
                world_img = np.array(data["data"][curr_key]["obs"]["agentview_rgb"]).copy()
                wrist_img = np.array(data["data"][curr_key]["obs"]["eye_in_hand_rgb"]).copy()

                # convert gripper -1 close, 0 stay, +1 open to 0 open, 1 close
                prev_gripper_act = 1
                gripper_act = actions[..., -1].copy()
                # sim runs continuous grasp -> only grasp when fully closed to match blocking
                # if cfg.blocking_control:
                #     not_quite_done_yet = np.where(np.abs(gripper[1:,0] - gripper[:-1,0]) > 1e-3)
                #     gripper_act[not_quite_done_yet] = 0
                for i in range(len(gripper_act)):
                    gripper_act[i] = prev_gripper_act if gripper_act[i] == 0 else gripper_act[i]
                    prev_gripper_act = gripper_act[i]
                gripper_act[np.where(gripper_act == 1)] = 0
                gripper_act[np.where(gripper_act == -1)] = 1
                actions[..., -1] = gripper_act
                
                lowdim_gripper = np.sum(gripper, axis=1)[:,None]

                dic = {
                    "lowdim_ee": np.concatenate((ee_pos, ee_euler, lowdim_gripper), axis=1),
                    "lowdim_qpos": np.concatenate((qpos, lowdim_gripper), axis=1),
                    "front_rgb": world_img,
                    "wrist_rgb": wrist_img
                }

                obs_keys = dic.keys()
                
                # if cfg.blocking_control:
                #     # compute actual deltas s_t+1 - s_t (keep gripper actions)
                #     actions_tmp = actions.copy()
                #     actions_tmp[:-1, ..., :6] = (
                #         dic["lowdim_ee"][1:, ..., :6] - dic["lowdim_ee"][:-1, ..., :6]
                #     )
                #     actions = actions_tmp[:-1]

                #     # remove last state s_T
                #     for key in obs_keys:
                #         dic[key] = dic[key][:-1]

                # create demo group
                demo_key = f"demo_{episodes}"
                demo_keys[split].append(demo_key)
                ep_data_grp = grp.create_group(demo_key)

                # compute shortest angle -> avoid wrap around
                actions[..., 3:6] = shortest_angle(actions[..., 3:6])

                # add action dataset
                ep_data_grp.create_dataset("actions", data=actions)

                # add done dataset
                dones = np.zeros(len(actions)).astype(bool)
                dones[-1] = True
                ep_data_grp.create_dataset("dones", data=dones)

                # create obs and next_obs groups
                ep_obs_grp = ep_data_grp.create_group("obs")

                # if "language_instruction" not in obs_keys:
                #     dic["language_instruction"] = ["pick up the cube"] * len(dic["lowdim_ee"])
                #     print("WARNING: 'language_instruction' not in dataset, adding template instruction!")

                # manually add the instruction
                instruction = extract_instruction_from_filename(dataset_path)
                dic["language_instruction"] = [instruction] * len(dic["lowdim_ee"])
                print(instruction)
                obs_keys = dic.keys()
                #for some reason I can't get get_lang_emb to import..? I have it pip installed
                lang_emb = get_lang_emb(dic["language_instruction"][0])
                ep_obs_grp.create_dataset("lang_embed", data=lang_emb)
                dic["language_instruction"] = np.array(dic["language_instruction"], dtype='S100')
                # add obs and next_obs datasets
                for obs_key in obs_keys:
                    obs = dic[obs_key]
                    if "_rgb" in obs_key:
                        # crop images for training
                        x_min, x_max, y_min, y_max = cfg.aug.camera_crop
                        obs = obs[:, x_min : x_max, y_min : y_max]
                        # resize images for training
                        obs = np.stack([cv2.resize(img, cfg.aug.camera_resize) for img in obs])
                    # if obs_key == "language_instruction":
                    #     lang_emb = get_lang_emb(obs[0])
                    #     lang_emb = np.tile(lang_emb, (len(obs), 1))
                    #     ep_obs_grp.create_dataset("lang_embed", data=lang_emb)
                    #     obs = np.array(obs, dtype='S100')

                    ep_obs_grp.create_dataset(obs_key, data=obs)

                ep_data_grp.attrs["num_samples"] = len(actions)

                episodes += 1

####################
    # haven't touched any of this code since get_lang_emb doesn't work. Most likely need to get rid of split
        # robomimic doesn't do splits, since it's just too hard, so they eval on their trained data
        # create mask dataset
        grp_mask.create_dataset(split, data=np.array(demo_keys[split], dtype="S"))


    # dummy metadata so robomimic is happy
    grp.attrs["episodes"] = episodes
    grp.attrs["env_args"] = '{"env_type":  "blub", "type": "blub"}'
    grp.attrs["type"] = "blub"

    if cfg.normalize_acts:
        print("Computing training statistics ...")
        actions = np.concatenate(
            [grp[demo_key]["actions"] for demo_key in demo_keys["train"]]
        )
        stats = {
            "action": {
                "min": actions.min(axis=0),
                "max": actions.max(axis=0),
            }
        }

        pickle.dump(stats, open(os.path.join(hdf5_path, "stats"), 'wb'))

        print("Normalizing actions ...")
        for split in cfg.splits:
            for demo_key in demo_keys[split]:
                actions = grp[demo_key]["actions"]
                actions = normalize(actions, stats["action"])
                grp[demo_key]["actions"][...] = actions

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["blocking_control"] = cfg.blocking_control

    f.close()

    print("Saved at: {}".format(hdf5_path))


if __name__ == "__main__":
    run_experiment()
###########################