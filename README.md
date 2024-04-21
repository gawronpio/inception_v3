# Inception - Inception V3 AI model

## Description

This package implements Inception V3 AI model in Tensorflow.

## Usage

from inception import Inception  
model = Inception()

### Parameters:
- **inputs** - (tuple | list | KerasTensor) Shape of input for tuple and list or input tensor for KerasTensor.
- **params** - (dict | None) Parameters for the model. If None default parameters will be used. Any other model than Inception can be created giving parameters for the model.
- **dense_end** - (bool) If True, the model will be created with a dense layer at the end. It is valid only when params == None.

### params structure

Params dictionary must contain *model* key with a tuple/list of tuples/lists with standard parameters for the model. There can be 
used non-standard parameter. Then it have to be passed as additional key in params dictionary.  
Non-standard parameter value in dictionary contains tuple/list with tuples/lists (1st level) with tuples/lists (2nd level) with 
standard parameters. If there are more than one tuple/list in 1st level, they will be concatenated.

### The standard parameter list:

 - ('a', activation)  
   Activation function  
   - 'a' - is parameter layer type - activation  
   - activation - (str) is activation function name, e.g. 'ReLU'

 - ('ap2', kernel, stride, padding)  
   Average pooling layer - AvgPool2d
   - 'ap2' - is parameter layer type - average pooling 2d  
   - kernel - (int | tuple) is kernel size  
   - stride - (int | tuple) is stride size  
   - padding - (str) is padding type ('same' or 'valid')
 
 - ('cba', out_channels, kernel, stride, padding, activation)  
   Complex convolutional layer - Conv2d + BatchNorm2d + activation  
   - 'cba' - is parameter layer type - convolutional + batch normalization + activation  
   - out_channels - (int) is number of output channels  
   - kernel - (int | tuple) is Conv2d kernel size  
   - stride - (int | tuple) is Conv2d stride size  
   - padding - (str) is Conv2d padding type ('same' or 'valid')  
   - activation - (str) is activation layer function name, e.g. 'ReLU'
 
 - ('ctba', out_channels, kernel, stride, padding, activation)  
   Complex convolutional layer - ConvTranspose2d + BatchNorm2d + activation  
   - 'ctba' - is parameter layer type - transpose convolutional + batch normalization + activation  
   - out_channels - (int) is number of output channels  
   - kernel - (int | tuple) is ConvTranspose2d kernel size  
   - stride - (int | tuple) is ConvTranspose2d stride size  
   - padding - (str) is ConvTranspose2d padding type ('same' or 'valid')  
   - activation - (str) is activation layer function name, e.g. 'ReLU'
 
 - ('d', units)  
    Dense layer - Linear  
    - 'd' - is parameter layer type - dense  
    - units - (int) is number of output units
 
 - ('f', )  
    Flatten layer  
    - 'f' - is parameter layer type - flatten

 - ('gap2', )  
    Global average pooling layer - AdaptiveAvgPool2d  
    - 'gap2' - is parameter layer type - global average pooling 2d

 - ('mp2', kernel, stride, padding)  
    Max pooling layer - MaxPool2d  
    - 'mp2' - is parameter layer type - max pooling 2d  
    - kernel - (int | tuple) is kernel size  
    - stride - (int | tuple) is stride size  
    - padding - (str) is padding type ('same' or 'valid')

### Model parameters example

```python
params = {
    'model': (
        ('cba', 32, 3, 1, 'same', 'ReLU'),
        ('cba', 32, 3, 1, 'same', 'ReLU'),
        ('mp2', 2, 2, 'same'),
        ('cba', 64, 3, 1, 'same', 'ReLU'),
        ('cba', 64, 3, 1, 'same', 'ReLU'),
        ('mp2', 2, 2, 'same'),
        ('m1', ),
        ('m1', ),
        ('f', ),
        ('d', 512),
        ('a', 'ReLU'),
        ('d', 10),
        ('a', 'Softmax')
    ),
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
    ),
}
```