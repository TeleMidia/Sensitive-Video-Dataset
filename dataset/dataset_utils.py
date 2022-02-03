import pandas as pd
import numpy as np
import subprocess as sp
from tqdm import tqdm
import os

def get_filename_no_ext(filepath):
    return filepath.split('/')[-1].split('.')[0]

def get_ext(filepath):
    return filepath.split('/')[-1].split('.')[-1]

def create_or_clean_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_contents = os.listdir(dir_path)
    for file in dir_contents:
        os.remove(os.path.join(dir_path,file))

def get_remote_path(path):
    if ':' in path:
        return path.split(':')
    else:
        return '', path

def sizeof_fmt(num, suffix='B'):
    for unit in ['Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def fits_into_category(df, main_tags):
    video_tags = df['tags']
    title = str(df['name']).lower()
    if (not is_nan(video_tags)) and (not is_nan(title)):
        video_tags = video_tags.split(',')
        for tag in main_tags:
            if tag in video_tags or tag in title:
                # print(f'Replaced "Unknown" with {tag}: {title}')
                return tag
    return 'Unknown'

def is_nan(x):
    if isinstance(x,(float,)) or x == None or x == np.nan:
        return True
    else:
        return False

def seconds_fmt(s):
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

def get_filename(filepath):
    return filepath.split('/')[-1]

def get_filesize(filename, du_list=None):
    for item in du_list:
        if filename == get_filename_no_ext(item):
            # if filename == '432906':
            #     print(filename, item)
            return item.split('\t')[0]

def get_id(filename):
    return filename.split('.')[0]

def get_filedir(filepath):
    return '/'.join(filepath.split('/')[:-1])

def get_parentdir(filepath):
    return filepath.split('/')[-2]

def get_macro_and_micro_classes_df(df):
    improper = df[df['macroclass'] == 'sensitive']
    proper = df[df['macroclass'] == 'safe']
    porn = df[df['microclass'] == 'pornography']
    gore = df[df['microclass'] == 'gore']
    return improper, proper, porn, gore

def sample_dataset(dataset, random_seed,pct=None,absolute=None, labels_column='main_tag',tolerance=4,min_count_for_small_tags=10):
    #tolerance = 4 # the value we add to make up for the rouding, comment to see its effect
    dt_numbers = dataset.groupby(['main_tag']).size().to_frame('count').sort_values('count', ascending=False)#.unstack()
    dt_numbers['percentage'] = dt_numbers['count']/len(dataset)
    max_p = dt_numbers['percentage'].max()
    dt_numbers['balanced'] = dt_numbers['percentage']/max_p
    if pct:
        dataset_pct = 0.1
        sample_size = np.ceil((dataset_pct*dt_numbers['count'].sum()) + tolerance)
    elif absolute:
        sample_size = absolute + tolerance
    else:
        raise('You must give either pct or absolute value for sampling!')
    limit_factor = (sample_size-(min_count_for_small_tags*len(dt_numbers.index)))/sum(dt_numbers['count']*dt_numbers['balanced'])
    print(f"Total Size: {dt_numbers['count'].sum()}, sample size: {sample_size}, limit factor: {limit_factor}")
    multiplier = dt_numbers['balanced'] * limit_factor
    #we gotta have at least the minimun amount of videos for each class
    dt_numbers['to_be_sampled'] = round(dt_numbers['count']*multiplier + min_count_for_small_tags)
    dt_numbers['to_be_sampled'] = dt_numbers['to_be_sampled'].astype(int)
    print(f"Actual sample size: {dt_numbers['to_be_sampled'].sum()}")
    dt_numbers[['to_be_sampled','count']].plot(kind='line',figsize=(25,5))
    #Actualy balancing the dataset
    dt_grouped = dataset.groupby('main_tag')
    df_list = []
    for (method, group) in dt_grouped:
        #print("{0:30s} shape={1}".format(method, group.shape))
        #print(f'{method}: {dt_numbers.loc[method,"new_count"]}')
        df_list.append(group.sample(n=dt_numbers.loc[method,'to_be_sampled'], random_state=random_seed))
    df_balanced = pd.concat(df_list)
    df_balanced_numbers = df_balanced.groupby(['main_tag']).size().to_frame('count').sort_values('count', ascending=False)#.unstack()
    dt_numbers['sample_count'] = df_balanced_numbers['count']
    dt_numbers[['sample_count','count']].plot(kind='line',figsize=(25,5))
    print(dt_numbers.head())
    return df_balanced

def print_verbose(text,verbose=True):
  if verbose: print(text)

def get_video_length_seconds(path, ssh_key_path=None):
    # https://stackoverflow.com/questions/15041103/get-total-length-of-videos-in-a-particular-directory-in-python
    # if you get error: UnicodeEncodeError: 'ascii' codec can't encode character '\xe9' in position 77: ordinal not in range(128)
    # check this question: https://stackoverflow.com/questions/47946134/subprocess-run-argument-encoding
    # basically you need utf8 encoding in your locale
    remote_server_name, filename = get_remote_path(path)
    command = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filename,
            ]
    if remote_server_name:  # if those are files on a remote server
        command = ["ssh", remote_server_name, "-i", ssh_key_path] + command
    try:
        filename = filename.encode('utf-8')
        result = sp.run(command,stdout=sp.PIPE,stderr=sp.PIPE, encoding='utf-8'
        )
        if result.stdout != b'':
            return float(result.stdout)#.encode('utf-8')
        else:
            print(f'Error in file: {filename}')
            print('SDOUT:, ', result.stdout)  # .encode('utf-8')
            print('SDERR:, ', result.stderr)
            return None
    except Exception as e:
        print(f'Error in file: {filename}')
        print(str(e))
        print('SDOUT:, ', result.stdout)#.encode('utf-8')
        print('SDERR:, ', result.stderr)
        return None

def check_min_and_max_durations(big_df, duration_cap=3600,duration_min=5):
    long_videos = (big_df['video_duration_secs'] > duration_cap)
    short_videos = (big_df['video_duration_secs'] < duration_min)
    pct_long_videos = (long_videos.sum()/len(big_df))*100
    pct_short_videos = (short_videos.sum()/len(big_df))*100
    print(f'There are {long_videos.sum()} ({pct_long_videos:.2f}%) videos longer than {seconds_fmt(duration_cap)}')
    print(f'There are {short_videos.sum()} ({pct_short_videos:.2f}%) videos shorter than {seconds_fmt(duration_min)}')

def get_videos_length_on_df(df_line, ssh_key_path=None):
    if df_line['is_video']:
        df_line['video_duration_secs'] = get_video_length_seconds(df_line['filepath'], ssh_key_path=ssh_key_path)
    return df_line

def statistics_gobbler(dataframes, names, no_fmt=False, short=False):
    dfs = []
    for i,df in enumerate(dataframes):
        if len(df) == 0:
            print(f'Dataframe: {names[i]} has size 0!')
            continue
        if no_fmt:
            stats = {
                'Video Count': [len(df)],
                'Total Duration': [df['video_duration_secs'].sum()],
                'Mean Duration': [df['video_duration_secs'].mean()],
                'STD Duration': [df['video_duration_secs'].std()],
                'Max Duration': [df['video_duration_secs'].max()],
                'Min Duration': [df['video_duration_secs'].min()],
                'Total Size': [df['filesize'].dropna().astype(int).sum()],
                'Mean Size': [df['filesize'].dropna().astype(int).mean()],
                'STD Size': [df['filesize'].dropna().astype(int).std()],
                #'Features Size': [df['filesize_features'].dropna().astype(int).sum()],
                'Tag coverage': [(~df['tags'].isnull()).sum()],
                'Tag coverage (%)': [((~df['tags'].isnull()).sum() / len(df)) * 100],
                # 'Main Tag existance': [],
            }
        else:
            if not short:
                stats = {
                    'Video Count': [len(df)],
                    'Total Duration': [seconds_fmt(df['video_duration_secs'].sum())],
                    'Mean Duration': [seconds_fmt(df['video_duration_secs'].mean())],
                    'STD Duration': [seconds_fmt(df['video_duration_secs'].std())],
                    'Max Duration': [seconds_fmt(df['video_duration_secs'].max())],
                    'Min Duration': [seconds_fmt(df['video_duration_secs'].min())],
                    'Total Size': [sizeof_fmt((df['filesize'].dropna().astype(int)).sum())],
                    'Mean Size': [sizeof_fmt((df['filesize'].dropna().astype(int)).mean())],
                    'STD Size': [sizeof_fmt((df['filesize'].dropna().astype(int)).std())],
                    #'Features Size': [sizeof_fmt((df['filesize_features'].dropna().astype(int)).sum())],
                    'Tag coverage': [(~df['tags'].isnull()).sum()],
                    'Tag coverage (%)': [((~df['tags'].isnull()).sum() / len(df)) * 100],
                    # 'Main Tag existance': [],
                }
            else:
                stats = {
                    'Video Count': [len(df)],
                    'Total Duration': [seconds_fmt(df['video_duration_secs'].sum())],
                    'Mean Duration': [seconds_fmt(df['video_duration_secs'].mean())],
                    'STD Duration': [seconds_fmt(df['video_duration_secs'].std())],
                    'Total Size': [sizeof_fmt((df['filesize'].dropna().astype(int)).sum())],
                    'Tag coverage (%)': [((~df['tags'].isnull()).sum() / len(df)) * 100],
                    # 'Main Tag existance': [],
                }
        dfs.append(pd.DataFrame(stats, index=[names[i]]))#, columns=[]
    return pd.concat(dfs).T


def glob_and_collect_statistics(path, ssh_key_path = None, recursive=False, verbose=False, level=0):
    """glob videos/files independetely of it is on the local machine or on a remote server"""
    remote_server_name, remote_path = get_remote_path(path)
    dir_size = 0
    df = pd.DataFrame()
    filetypes = ['mp4', 'webm']  # the file types supported

    # Getting all files via remote ls
    command = ["ls", "-p", remote_path.encode('utf-8')]
    if remote_server_name:  # if those are files on a remote server
        # ssh: -p, append / indicator to directories; -i path to indentify file; -d show full path
        command = ["ssh", remote_server_name, "-i", ssh_key_path] + command

    ls = sp.run(command, stdout=sp.PIPE,stderr=sp.PIPE, encoding='utf-8')
    out = ls.stdout.split('\n')

    directories = [dir for dir in out if dir != '' and dir[-1] == '/']
    files = [dir for dir in out if dir != '' and dir[-1] != '/']
    videos = [remote_path + file for file in files if
              file.split('/')[-1].split('.')[-1] in filetypes]
    non_videos = [remote_path + file for file in files if
                  file.split('/')[-1].split('.')[-1] not in filetypes]
    if remote_server_name:  # if those are files on a remote server
        videos_paths = [remote_server_name + ":" + filepath for filepath in videos]
        non_video_paths = [remote_server_name + ":" + filepath for filepath in non_videos]
    else:
        videos_paths = videos
        non_video_paths = non_videos
    df['filepath'] = videos_paths + non_video_paths
    df['is_video'] = len(videos) * [True] + len(non_videos) * [False]
    df['filename'] = df['filepath'].apply(get_filename_no_ext)
    df['parentdir'] = df['filepath'].apply(get_parentdir)

    # Try to get a overall size dir with du
    command = ["du", "-s", remote_path.encode('utf-8')]
    if remote_server_name:  # if those are files on a remote server
        command = ["ssh", remote_server_name, "-i", ssh_key_path] + command
    du_dir_out = sp.run(command, stdout=sp.PIPE,stderr=sp.PIPE, encoding='utf-8').stdout.split('\t')[0]
    if du_dir_out != '':
        dir_size = int(du_dir_out)

    # Priting progress and data
    tab = level * '\t'
    if verbose:
        print(100 * '-', '\n')
        print(f'{tab}#{remote_path}')
        print(f'{tab}Files found: {len(files)}')
        print(f'{tab}Videos found: {len(videos)}')
        print(f'{tab}Non-Videos found: {len(non_videos)}')
        print(f'{tab}Dirs found: {len(directories)}')
        print(f'{tab}Dir size: {sizeof_fmt(dir_size)}')

    # Getting the duration of each video on the dir
    print(f'{tab}Collecting video durations...')
    if len(videos) > 0:
        df.loc[df['is_video'] == True, 'video_duration_secs'] = df.loc[df['is_video'] == True].progress_apply(lambda row: get_videos_length_on_df(row, ssh_key_path=ssh_key_path), axis=1)
        df['video_duration_secs'] = df['video_duration_secs'].astype(float) # dá erro quando convertendo nan pra int, pois nan é float!

    # Getting disk usage of each file in the dir
    print(f'{tab}Collecting file sizes...')
    command = ["du", "-a", "--", remote_path.encode('utf-8')]
    if remote_server_name:  # if those are files on a remote server
        command = ["ssh", remote_server_name, "-i", ssh_key_path] + command
    du = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf-8')
    du_out = du.stdout.split('\n')
    df['filesize'] = df['filename'].progress_apply(get_filesize, args=([du_out]))
    df['filesize'] = df['filesize'].astype(int)

    print(f'{tab}Done! Recursing on child dirs...')
    if recursive:
        for directory in directories:
            if remote_server_name:  # if those are files on a remote server
                new_path = remote_server_name + ":" + os.path.join(remote_path, directory)
                a, b, c, d = glob_and_collect_statistics(ssh_key_path=ssh_key_path, path=new_path, recursive=recursive, verbose=verbose, level=level + 1)
            else:
                new_path = os.path.join(remote_path, directory)
                a, b, c, d = glob_and_collect_statistics(path=new_path, recursive=recursive, verbose=verbose, level=level + 1)
            videos_paths += a
            non_video_paths += b
            dir_size += c
            df = pd.concat([df, d], axis=0)
    return videos_paths, non_video_paths, dir_size, df
