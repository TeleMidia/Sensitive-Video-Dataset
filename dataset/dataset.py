import random, os, math, sys
from glob import glob
import argparse
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
sys.path.append('../../models/')
from utils import print_verbose
from dataset_utils import get_filename
import numpy as np

class Dataset:
    def __init__(self, collect_files=False, proper_path=None, improper_path=None, batch_size=1000, seed = 69420, verbose=False,tqdm=True, base_dir='./'):
        """
        Generates dataset.
        """
        self.proper_path = proper_path
        self.proper_df = None
        self.improper_df = None
        self.dataset = None
        self.improper_path = improper_path
        self.batch_size = batch_size
        self.batch_list = []
        self.base_dir = base_dir
        self.seed = seed
        self.verbose = verbose
        self.tqdm = tqdm
        random.seed(self.seed)
        if collect_files:
            self.proper_df, self.improper_df= self.collect_files_to_dataframe()
            self.proper_df, self.improper_df = self.balance_dataset()

            self.dataset = pd.concat([self.proper_df,self.improper_df], axis=0, sort=False)

            with open(self.base_dir+'/pickles/dataset.pkl', 'wb') as output:
                pickle.dump(self.dataset, output, pickle.HIGHEST_PROTOCOL)
            
        else:
            print('Loading dataset pickle...')
            with open(self.base_dir+'/pickles/dataset.pkl', 'rb') as input:
                self.dataset = pickle.load(input)
        self.max_sequence_size = Dataset.get_max_samples_count(self.dataset)
        print('Creating batches...')
        self.batch_list = self.create_batches()

    def load_and_make_class_dataframe(filepaths, save_pickle_path=None):
        df_list = []
        try:
            for filepath in tqdm(filepaths):
                df = pd.read_csv(filepath,index_col=0)
                df['filename'] = get_filename(filepath)
                df = df.reset_index()
                df = df.rename(columns={"index": "sample_order"})
                df_list.append(df)
            big_df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
            big_df = big_df.set_index(['filename','sample_order'])
            if save_pickle_path is not None:
                with open(save_pickle_path, 'wb') as output:
                    pickle.dump(big_df, output, pickle.HIGHEST_PROTOCOL)
            return big_df
        except Exception as e:
            print_verbose(e)
        return None

    def collect_files_to_dataframe(self):
        """
        Collects all single csv files and group them all by class in dataframes.
        Also generates a single dataframe with both classes.
        """
        if not os.path.exists(self.base_dir+'/pickles'): os.mkdir(self.base_dir+'/pickles')
        
        if self.proper_path:
            proper_paths = glob(self.proper_path+'/*')
            print_verbose(f'Loading {len(proper_paths)} proper files', self.verbose)
            proper_df = Dataset.load_and_make_class_dataframe(proper_paths,
                save_pickle_path=self.base_dir+"/pickles/dataset-proper.pkl")
        else:
            print('Loading old proper pkl...')
            with open(self.base_dir+'/pickles/dataset-proper.pkl', 'rb') as input:
                proper_df = pickle.load(input)

        if self.improper_path:
            improper_paths = glob(self.improper_path+'/*')
            print_verbose(f'Loading {len(improper_paths)} improper files', self.verbose)
            improper_df = Dataset.load_and_make_class_dataframe(improper_paths,save_pickle_path=self.base_dir+'/pickles/dataset-improper.pkl')
        else:
            print('Loading old improper pkl...')
            with open(self.base_dir+'/pickles/dataset-improper.pkl', 'rb') as input:
                improper_df = pickle.load(input)

        return proper_df, improper_df

    def create_batches(self):
        '''
        Creates balanced batches. 50% proper, 50% improper videos in each batch.
        '''
        batch_list = []
        proper_df = self.dataset[self.dataset['class']=='proper'].copy()
        improper_df = self.dataset[self.dataset['class']=='improper'].copy() 
        proper_list = list(proper_df.index.get_level_values(0).unique())
        improper_list = list(improper_df.index.get_level_values(0).unique())

        if len(proper_list) != len(improper_list):
            print_verbose(f'\n\nWarning: two classes have different sizes! proper: {len(proper_list)} improper: {len(improper_list)}')
        n_result_batches = math.ceil(len(proper_list)/(self.batch_size//2))
        for i in tqdm(range(n_result_batches)):
            if len(proper_list) < self.batch_size//2:
                sample_list = proper_list
                sample = proper_df.loc[proper_list] 
            else:
                sample_list = random.sample(proper_list,self.batch_size//2)
                sample = proper_df.loc[sample_list] 

            print_verbose(f'Proper sample size: {len(sample)}', self.verbose)
            [proper_list.remove(i) for i in sample_list]
            batch = sample

            if len(improper_list) < self.batch_size//2:
                sample_list = improper_list
                sample = improper_df.loc[sample_list]
            else:
                sample_list = random.sample(improper_list,self.batch_size//2)
                sample = improper_df.loc[sample_list]

            print_verbose(f'Improper sample size: {len(sample)}', self.verbose)
            [improper_list.remove(i) for i in sample_list]
            batch = pd.concat([batch,sample])
            
            index = batch.index.levels[0].to_list()
            random.shuffle(index)
            batch = batch.reindex(index, level=0) # shuffle batch
            batch_list.append(batch)
            #print_verbose(f'Generated batch of size: {len(batch)}', self.verbose)
            #proper = remove_sub_array(proper,sample_proper)
        print_verbose(f'Generated {len(batch_list)} batches!', self.verbose)
        return batch_list


    def balance_dataset(self):
        #Porn id: 42624525
        #Yt id: nsfRvU28-Og
        proper_df = self.proper_df.copy()
        improper_df = self.improper_df.copy()
        proper_list = list(proper_df.index.get_level_values(0).unique()) # getting unique filenames
        improper_list = list(improper_df.index.get_level_values(0).unique())

        print_verbose('Balancing dataset!', self.verbose)
        print_verbose(f'Total len before balancing:{len(proper_list)+len(improper_list)}', self.verbose)

        print_verbose(f'We have {len(improper_list)} improper image features!', self.verbose)
        print_verbose(f'We have {len(proper_list)} proper image features!', self.verbose)

        smallest_len = min(len(improper_list),len(proper_list))
        biggest_len = max(len(improper_list),len(proper_list))
        amount_to_drop = biggest_len-smallest_len
        if len(improper_list) == smallest_len: # if improper are the smaller than we need to reduce the proper ones
            print_verbose(f'Dropping {amount_to_drop} proper videos to balance the classes!', self.verbose)
            proper_list = random.sample(proper_list,smallest_len) # Selecting random samples to drop
        else:
            print_verbose(f'Dropping {amount_to_drop} improper videos to balance the classes!', self.verbose)
            improper_list = random.sample(improper_list,smallest_len) # Selecting random samples to drop


        print_verbose(f'Now we have {len(improper_list)} improper image features!', self.verbose)
        print_verbose(f'Now we have {len(proper_list)} proper image features!', self.verbose)
        print_verbose(f'Total len after balancing:{len(proper_list)+len(improper_list)}', self.verbose)

        proper_df = proper_df.loc[proper_list] # CHECK IF SELECTING A FILENAME WE GET ALL SAMPLES FROM THAT FILENAME
        improper_df = improper_df.loc[improper_list] # CHECK IF SELECTING A FILENAME WE GET ALL SAMPLES FROM THAT FILENAME
        return proper_df, improper_df

    @staticmethod
    def from_sequence_to_single_features(file_features):
        # we aggregate the sequence into mean, median, std, min and max
        columns_names_mean = [('mean_'+str(i)) for i in range(1152)]
        columns_names_median = [('median_' + str(i)) for i in range(1152)]
        columns_names_std = [('std_'+str(i)) for i in range(1152)]
        columns_names_min = [('min_' + str(i)) for i in range(1152)]
        columns_names_max = [('max_' + str(i)) for i in range(1152)]
        columns_names = columns_names_mean + columns_names_median + columns_names_std + columns_names_min + columns_names_max
        mean = np.expand_dims(np.mean(file_features,axis=0), axis=0)#.reshape((1, 1152))
        median = np.expand_dims(np.median(file_features,axis=0), axis=0)#.reshape((1, 1152))
        std = np.expand_dims(np.std(file_features,axis=0), axis=0)#.reshape((1, 1152))
        min = np.expand_dims(np.min(file_features,axis=0), axis=0)#.reshape((1, 1152))
        max = np.expand_dims(np.max(file_features,axis=0), axis=0)#.reshape((1, 1152))
        values = np.concatenate((mean,median,std,min,max),axis=1)
        new_batch = pd.DataFrame(values,columns=columns_names)
        new_batch = new_batch.fillna(0)
        return new_batch

    @staticmethod
    def get_size_of_sequence(x):
        return x.shape[0]

    @staticmethod
    def get_max_samples_count(batch):
        max_seq_size = 0
        for key, value in batch.items():
            if max_seq_size < value.shape[0]:
                max_seq_size = value.shape[0]
        return max_seq_size

    @staticmethod
    def get_max_samples_count_all_batches(batch_list):
        max_seq_size = 0
        for batch in tqdm(batch_list):
            value = Dataset.get_max_samples_count(batch)
            if max_seq_size < value:
                max_seq_size = value
        return max_seq_size

    @staticmethod
    def stdz_sequence(x):
        x.index = x.index.droplevel('filename')
        x, _ = Dataset.standardization(x,x.columns[:-1])
        return x

    @staticmethod
    def pad_sequence(sequence, max_size):
        seq_len = sequence.shape[0]
        pad_size = max_size - seq_len
        padded_seq = np.concatenate((sequence, np.zeros((pad_size, sequence.shape[1]))), axis=0)
        return padded_seq, seq_len

    @staticmethod
    def pad_batch(batch, max_sequence_size):
        seq_len_batch = []
        new_batch = []
        for i,seq in enumerate(batch.values()):
            padded_seq, seq_len = Dataset.pad_sequence(seq, max_sequence_size)
            seq_len_batch.append(seq_len)
            new_batch.append(padded_seq)
        return new_batch, seq_len_batch

    @staticmethod
    def reshape_features_for_RNN(features, labels):
        # Reshaping features to the format [batch size x max length x features]
        # The labels dont need any reshaping, they are good as [batch size x number of samples]
        # features = np.asarray(features)
        # labels = np.asarray(labels)
        # display(features.shape) # Should be somthing like (20, 665, 1152)
        # display(labels.shape) # Should be somthing like (13300, 2)
        return np.asarray(features), np.asarray(labels)

    @staticmethod
    def count_microclasses(batch, microclass):
        n_microclasses = 0
        for key, value in batch.items():
            label = key.split('_')[0]
            if label == 'improper': # if the microclass is not all, then it must be one of the improper micro classes
                if (microclass == 'gore') and 'gore' in key.split('_'):
                        n_microclasses = n_microclasses + 1
                elif (microclass == 'porn') and ('gore' not in key.split('_')):
                        n_microclasses = n_microclasses + 1
        return n_microclasses

    @staticmethod
    def batch_labels_and_values_from_npz(batch, labels_dummies=True, zero_audio=False, zero_image=False, microclass='all'):
        """
        This function is used only by the sequential models! Use get_labels_one_hots and pad_batch for sequential models
        """
        features_list = []
        labels_list = []
        count_proper = 0
        if microclass != 'all':
            n_microclass = Dataset.count_microclasses(batch, microclass)
            print(f'Number of {microclass}: {n_microclass}/{len(batch)}')
        for key, value in batch.items():
            # print(key, value.shape)
            label = key.split('_')[0]
            if zero_audio:
                for i in range(5):  # For each mean,median,std,min,max
                    start = 1024 + (i * 1152)
                    end = start + 128
                    #print(f'Start: {start}, End: {end}')
                    value[start:end] = 0.0
            if zero_image:
                for i in range(5): # For each mean,median,std,min,max
                    start = i * 1152
                    end = start + 1024
                    #print(f'Start: {start}, End: {end}')
                    value[start:end] = 0.0
            if (microclass != 'all'):
                if (label == 'improper'):
                    if (microclass == 'gore') and ('gore' not in key.split('_')):
                        continue
                    elif (microclass == 'porn') and ('gore' in key.split('_')):
                        continue
                if (label == 'proper'):
                    if n_microclass<=count_proper:
                        continue
                    else:
                        count_proper = count_proper + 1
            labels_list.append(label)
            features_list.append(value)
        if labels_dummies:
            labels = pd.get_dummies(labels_list)
        else:
            labels = labels_list
        #print('Order of labels: ', labels.columns) # ['improper', 'proper']
        # if (microclass != 'all'):
        #     print(f'Collected {count_proper} proper videos and {n_microclass} {microclass} videos!')
        #     print(f'Total before microclass selection: {len(batch)}, now: {n_microclass + count_proper}')
        return np.asarray(labels), np.concatenate(features_list)

    @staticmethod
    def get_labels_one_hots(batch):
        # Separating labels and features columns
        labels = [key.split('_')[0] for key in batch.keys()]
        labels = pd.get_dummies(labels)
        #print('Order of labels: ', labels.columns) # ['improper', 'proper']
        return np.asarray(labels)

    @staticmethod    
    def unstandardization(df, norm_params, features=None):
        if features is None:
            features = norm_params.index
        df = df.copy()
        for c in features:
            if c in df.columns:
                df.loc[:, [c]] = (df[[c]] * norm_params.loc[[c], 'std']) + norm_params.loc[[c], 'mean']
        return df

    @staticmethod    
    def standardization(df, features, norm_params=None):
        df = df.copy()
        if norm_params is None:
            norm_params = pd.DataFrame()
            norm_params['mean'] = df.loc[:, features].mean()
            norm_params['std'] = df.loc[:, features].std()
        for c in features:
            std = norm_params.loc[[c], 'std']
            std = std.replace(to_replace=0., value=1.)  # Avoid division by zero
            df.loc[:, [c]] = (df[[c]] - norm_params.loc[[c], 'mean']) / std

        return df, norm_params

    @staticmethod
    def quantize_features(clipped_embeddings, min_clip_value=-4.0, max_clip_value=4.0):
        """ This was extracted from the qunatize funcion from the audiovgg feature extraction"""
        assert len(clipped_embeddings.shape) == 2, (
                'Expected 2-d batch, got %r' % (clipped_embeddings.shape,))
        # Quantize by:
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = (
                (clipped_embeddings - min_clip_value) *
                (255.0 /
                 (max_clip_value - min_clip_value)))
        # - cast 8-bit float to uint8
        quantized_embeddings = np.around(quantized_embeddings).astype(np.uint8)
        return quantized_embeddings

    @staticmethod
    def clip_features(features, min_clip_value=-4.0, max_clip_value=4.0):
        """ This was extracted from the qunatize funcion from the audiovgg feature extraction"""
        assert len(features.shape) == 2, (
                'Expected 2-d batch, got %r' % (features.shape,))
        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = np.clip(features, min_clip_value, max_clip_value)
        return clipped_embeddings

    @staticmethod
    def standardization_min_max(df, features):
        df = df.copy()
        norm_params = pd.DataFrame()
        norm_params['min'] = df.loc[:, features].min()
        norm_params['max'] = df.loc[:, features].max()
        print(norm_params)
        for c in features:
            df.loc[:, [c]] = ((df[[c]] - norm_params.loc[[c], 'min']) / (
                    norm_params.loc[[c], 'max'] - norm_params.loc[[c], 'min'])).fillna(0.)
        return df, norm_params

    @staticmethod    
    def standardization_min_max_scaler(df, features):
        df = df.copy()
        for c in features:
            # Create a minimum and maximum processor object
            min_max_scaler = MinMaxScaler()
            # Create an object to transform the data to fit minmax processor
            x_scaled = min_max_scaler.fit_transform(df[c])
            # Run the normalizer on the dataframe
            df.loc[:, [c]] = x_scaled
        return df

if __name__ == "__main__":
    ##########
    # PARAMS #
    ##########
    #python3.6 dataset.py --improper /mnt/backup/VMR/dataset_extracted/improper/improper_embeddings --proper /mnt/backup/VMR/dataset_extracted/proper/proper_embeddings --batch_size 1000

    #python3.6 dataset.py --proper '/mnt/backup/vmr/dataset_extracted/proper/proper_embeddings/' --improper '/mnt/backup/vmr/dataset_extracted/improper/improper_embeddings/' --batch-size 1000
    parser = argparse.ArgumentParser()
    parser.add_argument('--proper',type=str,help='proper files path',required=False)
    parser.add_argument('--improper',type=str,help='improper files path',required=False)
    parser.add_argument('--batch-size',type=str,help='batch size',required=False)
    args = parser.parse_args()
    #print(args.proper,args.improper)
    dt = Dataset(proper_path=args.proper,improper_path=args.improper,batch_size=args.batch_size,collect_files=True)