import os

os.environ['PYTHONHASHSEED'] = str(69)
from feature_extractor_audio import extract_audio_features
from feature_extractor_image import extract_image_features
import subprocess as sp
from sys import argv
import numpy as np
import argparse
import glob
from tqdm import tqdm
import sys
import pandas as pd
import random
import tensorflow as tf


def reset_random_seeds(SEED=69):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


##LOCKING SEEDS
reset_random_seeds()

sys.path.append('../models/')
from utils import tensorflow_shutup, extract_and_save_wav, print_verbose, get_filename, get_filedir

tensorflow_shutup()

AUDIO_EXTRACTOR_PATH = "./"
IMAGE_EXTRACTOR_PATH = "./"

tag_map = {"proper": np.array([1, 0]), "improper": np.array([0, 1])}

##########
# PARAMS #
##########
PARAMS_PATH = os.path.join(os.getenv('HOME'), 'yt8m')
CHECKPOINT = f"{PARAMS_PATH}/vggish_model.ckpt"
PCA_PARAMS = f"{PARAMS_PATH}/vggish_pca_params.npz"
SSH_USER = 'pedropva'
KEY_PATH = f'/home/{SSH_USER}/.ssh/id_rsa'
TEMP_PATH = "/mnt/backup/vmr/temp/"
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)


# def find_file_in_list(file, list):
#     #l = list.copy()
#     #l.sort()
#     for i in range(len(list)):
#         if file == list[i]:
#             return True
#     print(f'Didnt find it in the list, but python in says: is is in the list? {file in list}')
#     print(file)
#     return False

################
# FILE CHECKER #
################
def check_extracted(filepaths, tag, save_path):
    new_filepaths = []
    remote_server_name, remote_path = get_remote_path(save_path)
    if remote_server_name:
        ls = sp.run(["ssh", remote_server_name, "-i", KEY_PATH, "ls", "-p", remote_path], stdout=sp.PIPE,
                    stderr=sp.PIPE)
        out = ls.stdout.decode("utf-8").split('\n')
        remote_extracted_videos = [file.split('.')[0].lower().strip() for file in out if
                  file != '' and file[-1] != '/']

    set_filepaths = set(filepaths)
    if len(filepaths) - len(set_filepaths):
        print(f'Watch out there are {len(filepaths) - len(set_filepaths)} duplicates on the source files!')

        # here only we want the ones that were resquested but are not extracted, we cant do anything about otherwise


    print("Checking for which files have not been extracted yet...")
    for file in tqdm(set_filepaths):
        file_id = file.split('/')[-1].split('.')[0].lower().strip()

        if remote_server_name:
            set_remote_extracted_videos = set(remote_extracted_videos)
            if len(remote_extracted_videos) - len(set_remote_extracted_videos):
                print(
                    f'Watch out there are {len(remote_extracted_videos) - len(set_remote_extracted_videos)} duplicates on the source files!')

            if file_id not in set_remote_extracted_videos:
                new_filepaths.append(file)
        else:
            if not os.path.exists(f"{save_path}{file_id}.csv"):
                new_filepaths.append(file)
    print(f'There were {len(set_filepaths) - len(new_filepaths)} {tag} videos already processed!')
    print(f'There are {len(new_filepaths)}/{len(set_filepaths)} {tag} videos to process!')
    return new_filepaths


def labels_and_features_to_csv(video_features, tag, where_to_save, file_name):
    try:
        remote_server_name, remote_path = get_remote_path(where_to_save)
        if remote_server_name:  # if those are files on a remote server
            where_to_save = TEMP_PATH

        df = pd.DataFrame(video_features)
        df['class'] = tag
        df.to_csv(f"{where_to_save}{file_name}.csv")

        if remote_server_name:  # if those are files on a remote server
            print_verbose(f"Uploading features", True)
            out = sp.run(["scp", '-i', KEY_PATH, f"{where_to_save}{file_name}.csv", f'{remote_server_name}:{remote_path}'], stdout=sp.PIPE, stderr=sp.PIPE)
            #print(out.stdout)
            #print(out.stderr)
            # removing the saved features
            os.remove(f"{where_to_save}{file_name}.csv")
        return True
    except Exception as e:
        print(e)
        return False


def load_video_features(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df


def get_remote_path(path):
    if ':' in path:
        return path.split(':')
    else:
        return None, None


def detect_tag(filename):
    if ('improper' in filename) or ('improprio' in filename) or ('porn' in filename):
        return 'improper'
    return 'proper'


def glob_videos(path, recursive=False, verbose=False):
    """glob videos independetely of it is on the local machine or on a remote server"""
    remote_server_name, remote_path = get_remote_path(path)
    filepaths = []
    filetypes = ['mp4', 'webm']  # the file types supported
    if remote_server_name:  # if those are files on a remote server
        # ssh: -p, append / indicator to directories; -i path to indentify file; -d show full path
        ls = sp.run(["ssh", remote_server_name, "-i", KEY_PATH, "ls", "-p", remote_path], stdout=sp.PIPE,
                    stderr=sp.PIPE)
        out = ls.stdout.decode("utf-8").split('\n')
        directories = [dir for dir in out if dir != '' and dir[-1] == '/']
        files = [dir for dir in out if dir != '' and dir[-1] != '/']
        videos = [remote_server_name + ":" + remote_path + file for file in files if
                  file.split('/')[-1].split('.')[-1] in filetypes]
        if verbose:
            print(f'Im on {remote_path}')
            print(f'Files found: {len(files)}')
            print(f'Videos found: {len(videos)}')
            print(f'Dirs found: {len(directories)}')
        filepaths = videos
        if recursive:
            for directory in directories:
                subdir_videos = glob_videos(remote_server_name + ":" + remote_path + '/' + directory, recursive, verbose)
                filepaths += subdir_videos
    else:  # if its local
        filetypes = ['*.'+i for i in filetypes]#['*.mp4', '*.webm']  # the file types supported
        for extension in filetypes:
            filepaths.extend(glob.glob(path + extension, recursive=recursive))
    return filepaths


################
# FILE MANAGER #
################
def file_manager(path, inverse, tag, recursive=False, destination=None, verbose=False):
    filepaths = glob_videos(path, recursive)
    print(f"{len(filepaths)} {tag} files found.")
    filepaths.sort(reverse=inverse)

    if destination:
        remote_server_name, remote_path = get_remote_path(destination)
        save_path = f"{destination}/{tag}_embeddings/"
    else:
        parent_dir = '/'.join(path.split('/')[:-2])
        save_path = f"{parent_dir}/{tag}_embeddings/"

    remote_server_name, remote_path = get_remote_path(save_path)
    if remote_server_name:
        pass
        # TODO FIGURE OUT HOW TO CREATE THIS DIR ON REMOTE DIR
    else:
        if not os.path.exists(save_path):  # needs to be adapted to work in a remote server
            os.makedirs(save_path)

    print(f'Saving {tag} embeddings in {save_path}')

    filepaths_filtered = check_extracted(filepaths, tag, save_path)

    for i, ffile in enumerate(filepaths_filtered):
        print_verbose("-" * 15, True)
        if tag == 'auto':
            tag = detect_tag(ffile)
        # if remote, download video to local, extract features, then delete video
        remote_server_name, remote_video_path = get_remote_path(ffile)
        if remote_server_name:
            if not os.path.exists(f"{TEMP_PATH}{ffile.split('/')[-1]}"):
                print_verbose(f"Downloading video {ffile.split('/')[-1]}, {i}/{len(filepaths_filtered)}", True)
                out = sp.run(["scp", '-i', KEY_PATH, ffile, TEMP_PATH], stdout=sp.PIPE, stderr=sp.PIPE)
                # print(out.stdout)
                # print(out.stderr)
            ffile = TEMP_PATH + (ffile.split('/')[-1])
        video_features = extract_features(ffile, tag)

        if isinstance(video_features, np.ndarray):
            if save_path:
                print_verbose(f'Saving {len(video_features)} samples...',verbose)
                if not labels_and_features_to_csv(video_features, tag, where_to_save=save_path, file_name=get_filename(ffile)):
                    print('Failed to save video features!')

            if remote_server_name:
                # removing the extracted video file
                os.remove(ffile)


#############
# EXTRACTOR #
#############
def extract_features(ffile, tag='proper', verbose=True):
    ##LOCKING SEEDS
    reset_random_seeds()
    # tag shold be "proper" or "improper"
    no_audio = False
    # reading files
    print_verbose(f"Reading {tag} file: {ffile}", verbose)
    file_name = get_filename(ffile)

    # extract frames features
    print_verbose(f"Extracting {tag} frames features", verbose)
    try:
        image_features = np.array(extract_image_features(ffile))
    except Exception as e:
        print('Error extracting frames features: ' + str(e))
        return None

    # print_verbose(f'frames features shape: {image_features.shape}',verbose)

    # extract audio from video
    if not extract_and_save_wav(ffile, TEMP_PATH + file_name + '.wav'):
        no_audio = True  # we set the audio features as 128 zeroed features

    # extract audio features
    print_verbose(f"Extracting {tag} audio features", verbose)
    # this needs to happpen after the video extraciton because of the feature len to put zeros
    if not no_audio:  # if the video has audio
        try:
            audio_features = np.array(
                extract_audio_features(CHECKPOINT, PCA_PARAMS, wav_file=f"{TEMP_PATH}{file_name}.wav"))
        except Exception as e:
            print('Error extracting audio features: ' + str(e))
            print('Putting zeros on the audio features!')
            audio_features = np.zeros((len(image_features), 128))
        # removing the extracted wav file
        os.remove(f'{TEMP_PATH}{file_name}.wav')
    else:
        audio_features = np.zeros((len(image_features), 128))

    print_verbose(f'Extracted {image_features.shape} image_features and {audio_features.shape} audio_features!',
                  verbose)

    if len(image_features) == 0 or len(image_features) == 0:
        print('Error extracting features. One of the medias is empty.')
        return None

    if audio_features.shape[0] > image_features.shape[0]:
        audio_image_diff = audio_features.shape[0] - image_features.shape[0]
        print(f'Correcting samples by dropping {audio_image_diff} samples of audio!')
        audio_features = audio_features[:-audio_image_diff]
    if audio_features.shape[0] < image_features.shape[0]:
        audio_image_diff = image_features.shape[0] - audio_features.shape[0]
        print(f'Correcting samples by dropping {audio_image_diff} samples of imaging!')
        image_features = image_features[audio_image_diff:]

    # aggregating audio and video
    try:
        video_features = np.hstack((image_features, audio_features))
    except Exception as e:
        print(f'image_features shape:{image_features.shape}')
        print(f'audio_features shape:{audio_features.shape}')
        print(e)
        quit('Error matching shapes!')
        return None

    return video_features


if __name__ == u'__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proper', type=str, help='proper files path, with a asterisc on the end!', required=False)
    parser.add_argument('--improper', type=str, help='improper files path, with a asterisc on the end!', required=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--inverse', type=bool, default=False)
    parser.add_argument('--recursive', type=bool, default=False)
    parser.add_argument('--destination', type=str, help='Where to send the resulting files.', default=None)
    # TODO NOT DELETE THE FILES WITH HAVE ERROR AUDIO OR ANY ERROR
    # TODO DEAL WITH CONCURRENCE?
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.proper:
        file_manager(args.proper, args.inverse, 'proper', args.recursive, args.destination)

    if args.improper:
        file_manager(args.improper, args.inverse, 'improper', args.recursive, args.destination)
