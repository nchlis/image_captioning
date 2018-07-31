The usage of each file is the following:

- load_images.py: reads the Flockr8K images, encodes then with a pre-trained ResNet50 and saves them to disk as Flickr8k_images_encoded.npy. The image filenames are saved into Flickr8k_images_filenames.npy.
- load_captions.py: reads the FLickr8K captions and saves then into captions.npy
- train_model_GRU.py, train_model_LSTM.py: They train a Merge architecture using a GRU and a LSTM, respectively
- train_model_GRU_inject.py, train_model_LSTM.py: same as above, but using Inject architecture
- evaluate_model.py: calculates BLEU scores for a given model
- plot_results.py: plots the results of all models (figures of the blog post comparing the methods)
- generate_new_caption.py: loads a trained model and generates captions for all images present in the ./captioned_images folder

