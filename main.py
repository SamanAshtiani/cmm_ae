
import numpy as np
import tensorflow as tf

def dataset():
    # Set seed for reproducibility
    np.random.seed(42)

    # Simulate scRNA-seq data with 2000 samples and 2000 genes
    num_samples = 2000
    num_genes = 2000

    return(np.random.normal(loc=0.0, scale=1.0, size=(num_samples, num_genes)))

def ae(num_genes = 2000):
    import tensorflow as tf
    # don't use gpu
    tf.config.set_visible_devices([], 'GPU')
    # Set seed for reproducibility
    import tensorflow as tf

    # Set seed for reproducibility
    tf.random.set_seed(42)

    # Define autoencoder architecture
    input_shape = (num_genes,)
    latent_dim = 1000  

    input_layer = tf.keras.layers.Input(shape=input_shape)
    encoder_layer = tf.keras.layers.Dense(latent_dim, activation='relu')(input_layer)
    decoder_layer = tf.keras.layers.Dense(num_genes, activation='linear')(encoder_layer)

    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder_layer)

    # Compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train autoencoder
    autoencoder.fit(scRNAseq_data, scRNAseq_data, epochs=10, batch_size=32)

    # Get encoder weights and transpose to get input layer weights
    encoder_weights = autoencoder.get_weights()[0]
    input_weights = encoder_weights.T


    print("encoder weights: ",encoder_weights.shape)
    print("data :", scRNAseq_data.shape)
    print("input_weights: ", input_weights.shape)


# test run for tf
def test_tf():
    #don't use gpu
    tf.config.set_visible_devices([], 'GPU')
    cifar = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100,)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64)

import bz2
import gzip
import lzma
import pickle

import brotli


class SomeObject():

    a = 'some data'
    b = 123
    c = 'more data'

    def __init__(self, i):
        self.i = i

#for thedee


#
#
# # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = [SomeObject(i) for i in range(1, 10)]
    print(data[0])
    # with open('no_compression.pickle', 'wb') as f:
    #     pickle.dump(data, f)
    #
    # with gzip.open("gzip_test.gz", "wb") as f:
    #     pickle.dump(data, f)
    #
    # with bz2.BZ2File('bz2_test.pbz2', 'wb') as f:
    #     pickle.dump(data, f)
    #
    # with lzma.open("lzma_test.xz", "wb") as f:
    #     pickle.dump(data, f)
    #
    # with open('no_compression.pickle', 'rb') as f:
    #     pdata = f.read()
    #     with open('brotli_test.bt', 'wb') as b:
    #         b.write(brotli.compress(pdata))


    # scRNAseq_data = dataset()
    #   print(scRNAseq_data[:5,:5])
    # test_tf()
    # ae()

#python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
#print(tf.__version__)
