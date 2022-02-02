# Sensitive-Content-Dataset

Massive amounts of video are uploaded on video-hosting platforms every minute. This volume of data presents a challenge in controlling the type of content uploaded to these video hosting services, for those platforms are responsible for any sensitive media uploaded by their users.
There has been an abundance of research on methods for developing automatic detection of sensitive content. In this paper, we present a sensitive video dataset for binary video classification (whether there is sensitive content in the video or not), containing 127 thousand tagged videos, Each with their extracted audio and visual embeddings.  
We define sensitive content as sex, violence, gore or any media that may cause distress on the viewer.

## Objective
To foster new methods for sensitive content detection in video.
## Feature/embeddings extractor
We extracted visual and audio embeddings, concatenated them and save the labels and features of each video.
The image features were extracted by [Inception V3](https://github.com/google/youtube-8m/tree/master/feature_extractor), the output of this network is 1024 floats.
The audio embeddings were extracted by the network [Vggish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), the output of this network is 128 floats.

The dataset has two variations: 
- Sequential: Each sample remains as it was extracted, a single video generates a sequence of N samples. In this variation, inside each npz file each instance is represented by a N by 1152 numpy array.
- Non-Sequential: All samples of a video are aggregated into a single sample, resulting in each instance having a shape of 1 by 5760, this single sample summarizes the entire video.

After the features of all the videos were extracted, we split a test portion (not public) and batched all the videos.

Each npz file represents a batch of variable size, but all split to have at max 4 Gbs when loaded to memory.
Each npz file has keys and values, the keys are string in the format (label)\_(video id). Some examples of keys in the npz file: "improper_29024487", "proper_MqnZqzAxQTk", "improper_gore122". Videos labeled as "improper" are Sensitive, and "proper" are safe.

The values are the videos features stored in numpy arrays, of varying shape, depending on the dataset variation (sequential or non-sequential).

There is also a main dataframe (we called it index), this dataframe is indexed by video id and contains all the other gathered data, such as tags, file size, duration, and title.

## Data
### Sensitive content
#### Pornography
Videos collected from public Xvideos database.
### Gore
Violence and blood in general collected in multiple violent content websites.

## Safe content
#### Youtube videos
Videos collected from the [youtube8m](https://research.google.com/youtube8m/) dataset.

## Dataset tree
<pre>
  .
  ├── dataset_index.csv # The main dataframe, indexed by video id and contains all the gathered data, such as tags, file size, duration, and title.
  ├── train_val_batches # This directory holds the sequential data (videos as times-series) batches for training/validation
  │   ├── batch_0.npz # An equally balanced (Safe x Sensitive) batch containing 407 instances
  │   ├── batch_1.npz # An equally balanced (Safe x Sensitive) batch containing 407 instances
  .   .   .   ...
  │   └── batch_211.npz # An equally balanced (Safe x Sensitive) batch containing 407 instances
  │   └── unbalanced_batch_0.npz # An unbalanced batch containing the remaining instances
  ├── non_sequential_train_val_batches # This directory holds the non-sequential (whole video represented by a single embedding) data batches for training/validation
  │   ├── batch_0.npz # An equally balanced (Safe x Sensitive) batch containing 407 instances
  │   ├── batch_1.npz # An equally balanced (Safe x Sensitive) batch containing 407 instances
  .   .   .   ...
  │   └── batch_211.npz # An equally balanced (Safe x Sensitive) batch containing 407 instances
  │   └── unbalanced_batch_0.npz # An unbalanced batch containing the remaining instances
  └── README.md   # The file that describes the dataset repository
</pre>
