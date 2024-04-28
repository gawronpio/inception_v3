"""
This file contains the model parameters for the Inception v3 model.
The model is defined as a list of tuples, where each tuple represents a layer.
The first element of the tuple is the layer type, and the rest of the elements are the layer parameters.
The layer types are as follows:
'a' - Activation
    (layer_type, activation, optional - dictionary with activation parameters
'ap2' - AveragePooling2D
    (layer_type, pool_size, strides, padding)
'cba' - Conv2D, BatchNormalization, Activation
    (layer_type, filters, kernel, strides, padding)
'd' - Dense
    (layer_type, units)
'f' - Flatten
    (layer_type, optional - channels after flatten)
'gap2' - GlobalAveragePooling2D
'mp2' - MaxPooling2D
    (layer_type, pool_size, strides, padding)

There are also some special layer types that are used to define the Inception v3 modules. These layers have to be defined in the 'model_param' dictionary as separate items.
'm1' - Inception v3 module 1
'm1r' - Inception v3 module 1 grid reduction
'm2' - Inception v3 module 2
'm2r' - Inception v3 module 2 grid reduction
'm3' - Inception v3 module 3
'm3a' - Inception v3 module 3 auxiliary
"""

model_params = {
    'model': [
        # input: 2, 299, 299
        ('cba', 32, 3, 2, 'same', 'ReLU'),  # 32, 149, 149
        ('cba', 32, 3, 1, 'same', 'ReLU'),  # 32, 147, 147
        ('cba', 64, 3, 1, 'same', 'ReLU'),  # 64, 145, 145
        ('mp2', 3, 2, 'same'),  # 64, 73, 73
        ('cba', 80, 3, 1, 'same', 'ReLU'),  # 80, 71, 71
        ('cba', 192, 3, 2, 'same', 'ReLU'),  # 192, 35, 35
        ('cba', 288, 3, 1, 'same', 'ReLU'),  # 288, 33, 33
        ('m1', ),  # 288, 33, 33
        ('m1', ),  # 288, 33, 33
        ('m1', ),  # 288, 33, 33
        ('m1r', ),  # 768, 17, 17
        ('m2', ),  # 768, 17, 17
        ('m2', ),  # 768, 17, 17
        ('m2', ),  # 768, 17, 17
        ('m2', ),  # 768, 17, 17
        ('m2', ),  # 768, 17, 17
        ('m2r', ),  # 1280, 9, 9
        ('m3', ),  # 2048, 9, 9
        ('m3', ),  # 2048, 9, 9
        # output 2048, 9, 9
    ],
    'm1': (
        (
            ('cba', 64, 1, 1, 'same', 'ReLU'),
            ('cba', 96, 3, 1, 'same', 'ReLU'),
            ('cba', 96, 3, 1, 'same', 'ReLU'),
        ),
        (
            ('cba', 48, 1, 1, 'same', 'ReLU'),
            ('cba', 64, 3, 1, 'same', 'ReLU'),
        ),
        (
            ('ap2', 3, 1, 'same'),
            ('cba', 64, 1, 1, 'same', 'ReLU'),
        ),
        (
            ('cba', 64, 1, 1, 'same', 'ReLU'),
        ),
    ),
    'm1r': (
        (
            ('cba', 64, 1, 1, 'same', 'ReLU'),
            ('cba', 96, 3, 1, 'same', 'ReLU'),
            ('cba', 96, 3, 2, 'same', 'ReLU'),
        ),
        (
            ('cba', 384, 3, 2, 'same', 'ReLU'),
        ),
        (
            ('ap2', 3, 2, 'same'),
            ('ap2', 1, 1, 'same'),
        ),
    ),
    'm2': (
        (
            ('cba', 128, 1, 1, 'same', 'ReLU'),
            ('cba', 128, 1, 1, 'same', 'ReLU'),
            ('cba', 128, 3, 1, 'same', 'ReLU'),
            ('cba', 128, 1, 1, 'same', 'ReLU'),
            ('cba', 192, 3, 1, 'same', 'ReLU'),
        ),
        (
            ('cba', 128, 1, 1, 'same', 'ReLU'),
            ('cba', 128, 1, 1, 'same', 'ReLU'),
            ('cba', 192, 3, 1, 'same', 'ReLU'),
        ),
        (
            ('ap2', 3, 1, 'same'),
            ('cba', 192, 1, 1, 'same', 'ReLU'),
        ),
        (
            ('cba', 192, 1, 1, 'same', 'ReLU'),
        ),
    ),
    'm2r': (
        (
            ('cba', 192, 1, 1, 'same', 'ReLU'),
            ('cba', 192, 1, 1, 'same', 'ReLU'),
            ('cba', 192, 3, 1, 'same', 'ReLU'),
            ('cba', 192, 3, 2, 'same', 'ReLU'),
        ),
        (
            ('cba', 192, 1, 1, 'same', 'ReLU'),
            ('cba', 320, 3, 2, 'same', 'ReLU'),
        ),
        (
            ('ap2', 3, 2, 'same'),
            ('ap2', 1, 1, 'same'),
        ),
    ),
    'm3': (
        (
            ('cba', 448, 1, 1, 'same', 'ReLU'),
            ('cba', 384, 3, 1, 'same', 'ReLU'),
            ('m3a', ),
        ),
        (
            ('cba', 384, 1, 1, 'same', 'ReLU'),
            ('m3a', ),
        ),
        (
            ('ap2', 3, 1, 'same'),
            ('cba', 192, 1, 1, 'same', 'ReLU'),
        ),
        (
            ('cba', 320, 1, 1, 'same', 'ReLU'),
        ),
    ),
    'm3a': (
        (
            ('cba', 384, (1, 3), 1, 'same', 'ReLU'),
        ),
        (
            ('cba', 384, (3, 1), 1, 'same', 'ReLU'),
        ),
    ),
    'dense_end': (
        ('gap2',),  # 2048, 1, 1
        ('f',),  # 2048
        ('d', 1000),  # 1000
        ('a', 'ReLU'),  # 1000
    ),
}
