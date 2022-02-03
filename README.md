# Sensitive Video Dataset

Massive amounts of video are uploaded on video-hosting platforms every minute. This volume of data presents a challenge in controlling the type of content uploaded to these video hosting services. Those platforms are responsible for any sensitive media uploaded by their users. In this context, we propose the 110K Sensitive Video Dataset for binary video classification (whether there is sensitive content in the video or not), containing more than 110 thousand tagged videos. Additionally, we separated an exclusive subset with 11 thousand videos for testing in Kaggle.

To compose the sensitive video subset, we collected videos with content of sex, violence, and gore from various internet sources. While composing the subset of safe videos, we collect videos from everyday life, online courses, tutorials, sports, etc. It is worth mentioning that we were concerned about creating more challenging examples for each class. We collected sex videos with people wearing full-body clothes (e.g., latex and cosplay) for the sensitive video class. Moreover, we have collected videos that could be misclassified as sensitive for the safe videos class, such as MMA, breastfeeding, pool party, beach, and other videos with a higher amount of skin exposure.


## Objective

To foster new methods for sensitive content detection in video.


## Feature/embeddings extractor

We extracted visual and audio embeddings, concatenated them, and saved each video's labels and features. [Inception V3](https://github.com/google/youtube-8m/tree/master/feature_extractor) extracted the visual features, generating embeddings of 1024-d. The audio embeddings were extracted by the [Vggish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) network, generating embeddings of 128-d.

The dataset has two variations: 
- Sequential: Each video is sampled into windows of 0.96s, resulting in an array of shapes (N, 1152), where the video duration limits N.
- Non-Sequential: The entire video is globally aggregated as a single sample, generating an array of shapes (1, 1152). Additionally, we compute the mean, median, max, min, and std for each feature, resulting in a final array of shapes (1, 5760) for each video.

We structured this dataset into chunks with a max size of 4GB. Each chunk was stored as an NPZ file. Chunks are composed of keys and values; the keys are strings in the format (label)_(video id) (for instance, "improper_29024487", "proper_MqnZqzAxQTk", "improper_gore122"). Videos labeled as "improper" are Sensitive, and "proper" are safe. The values are the audio-visual embeddings stored as NumPy arrays. 

A CSV file in the root directory contains all video indexes, metadata, and tags.


## Data Collection

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

## Loading and displaying the data
```python
import numpy as np
import glob
import tqdm # Optional, just remove the tqdm usage if not using

batch_filepaths = glob.glob('./non_sequential_train_val_batches/*')
batch_filepaths.sort(key=os.path.getmtime)

for batch_filepath in tqdm(batch_filepaths):
    with np.load(batch_filepaths) as batch:
        for keys, values in batch.items():
            print(f'File: {keys}, shape: {values.shape}')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
We are also open to suggestions for new feature extractiom methods! If you want to suggest a new feature extraction method, please open an issue and we'll discuss it!

## Papers using this dataset

P. V. de Freitas, G. N. d. Santos, A. J. Busson, Á. L. Guedes, and S. Colcher,“A baseline for nsfw video detection in e-learning environments”, in Proceedings of the 25th Brazillian Symposium on Multimedia and the Web, 2019, pp.357–360. [Online].Available: https://doi.org/10.1145/3323503.3360625

P. V. de Freitas, A. J. Busson, Á. L. Guedes, and S. Colcher, “A deep learning approach to detect pornography videos in educational repositories”, in Anais do XXXI Simpósio Brasileiro de Informática na Educação.  SBC, 2020, pp.1253–1262. [Online].Available: https://sol.sbc.org.br/index.php/sbie/article/view/12881

A. C. Serra, P. R. C. Mendes, P. V. A. de Freitas, A. J. G. Busson, A. L. V.Guedes, and S. Colcher, “Should i see or should i go: Automatic detection of sensitive media in messaging apps”, in Proceedings of the Brazilian Symposium on Multimedia and the Web, ser. WebMedia ’21.  New York,NY, USA: Association for Computing Machinery, 2021, p. 229–236. [Online].Available: https://doi.org/10.1145/3470482.3479639

## Cite this dataset

Pedro Vinicius Almeida de Freitas, Gabriel Noronha Pereira dos Santos, Antonio José Grandson Busson, Alan Livio Vasconcelos Guedes, Sérgio Colcher, February 2, 2022, "110K Sensitive Video Dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/sx01-1p81. 

Bibtex
<pre>
@data{sx01-1p81-22,
doi = {10.21227/sx01-1p81},
url = {https://dx.doi.org/10.21227/sx01-1p81},
author = {Almeida de Freitas, Pedro Vinicius and Noronha Pereira dos Santos, Gabriel and José Grandson Busson, Antonio and Livio Vasconcelos Guedes, Alan and Colcher, Sérgio},
publisher = {IEEE Dataport},
title = {110K Sensitive Video Dataset},
year = {2022} } 
</pre>

## License
[GPL](https://choosealicense.com/licenses/gpl-3.0/)
