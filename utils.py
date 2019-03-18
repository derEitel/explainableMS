import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from sklearn.metrics import recall_score


# data loading
def load_nifti(file_path, dtype=np.float32, incl_header=False, z_factor=None, mask=None, zoom_mode="cubic"):
    """
    Loads a volumetric image in nifti format (extensions .nii, .nii.gz etc.)
    as a 3D numpy.ndarray.
    
    Args:
        file_path: absolute path to the nifti file
        
        dtype(optional): datatype of the loaded numpy.ndarray
        
        incl_header(bool, optional): If True, the nifTI object of the 
        image is also returned.
        
        z_factor(float or sequence, optional): The zoom factor along the
        axes. If a float, zoom is the same for each axis. If a sequence,
        zoom should contain one value for each axis.
        
        mask(ndarray, optional): A mask with the same shape as the
        original image. If provided then the mask is element-wise
        multiplied with the image ndarray
    
    Returns:
        3D numpy.ndarray with axis order (saggital x coronal x axial)
    """
    
    img = nib.load(file_path)
    struct_arr = img.get_data().astype(dtype)
    
    # replace infinite values with 0
    if np.inf in struct_arr:
        struct_arr[struct_arr == np.inf] = 0.
    
    # replace NaN values with 0    
    if np.isnan(struct_arr).any() == True:
        struct_arr[np.isnan(struct_arr)] = 0.
        
    if mask is not None:
        struct_arr *= mask
        
    if z_factor is not None:
        if zoom_mode == "cubic":
            struct_arr = zoom(struct_arr, z_factor, order=3)
        elif zoom_mode == "bilinear":
            struct_arr = zoom(struct_arr, z_factor, order=1)
        elif zoom_mode == "nearest":
            struct_arr = zoom(struct_arr, z_factor, order=0)
        else:
            struct_arr = zoom(struct_arr, z_factor, order=2)
    
    if incl_header:
        return struct_arr, img
    else:
        return struct_arr

# transforms

def normalize_float(x, is_3d_mri=True):
    """ 
    Function that performs max-division normalization on a `numpy.ndarray` 
    matrix. 
    """
    if is_3d_mri:
        assert(len(x.shape) >= 4)
    for i in range(x.shape[0]):
        x[i] /= np.max(x[i])
    return x


class IntensityRescale:
    """
    Rescale image itensities between 0 and 1 for a single image.

    Arguments:
        masked: applies normalization only on non-zero voxels. Default
            is True.
        on_gpu: speed up computation by using GPU. Requires torch.Tensor
             instead of np.array. Default is False.
    """

    def __init__(self, masked=True, on_gpu=False):
        self.masked = masked
        self.on_gpu = on_gpu

    def __call__(self, image):
        if self.masked:
            image = self.zero_masked_transform(image)
        else:
            image = self.apply_transform(image)

        return image

    def apply_transform(self, image):
        if self.on_gpu:
            return normalize_float_torch(image)
        else:
            return normalize_float(image)

    def zero_masked_transform(self, image):
        """ Only apply transform where input is not zero. """
        img_mask = image == 0
        # do transform
        image = self.apply_transform(image)
        image[img_mask] = 0.
        return image

# metrics
def specificity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)


def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1)


def balanced_accuracy(y_true, y_pred):
    spec = specificity(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)
    return (spec + sens) / 2


# Data augmentations
def sagittal_flip(batch):
    """ 
        Expects shape (None, X, Y, Z, C).
        Flips along the X axis (sagittal).
        
    """
    thresh = 0.5
    batch_augmented = np.zeros_like(batch)
    for idx in range(len(batch)):
        rand = np.random.uniform()
        if rand > thresh:
            batch_augmented[idx] = np.flip(batch[idx], axis=0)
        else:
            batch_augmented[idx] = batch[idx]
    return batch_augmented

def translate(batch):
    """ 
        Expects shape (None, X, Y, Z, C).
        Translates the X axis.
    """
    batch_augmented = np.zeros_like(batch)
    for idx in range(len(batch)):
        rand = np.random.randint(-2, 3)
        if rand < 0:
            batch_augmented[idx,-rand:] = batch[idx,:rand]
        elif rand > 0:
            batch_augmented[idx,:-rand] = batch[idx,rand:]
        else:
            batch_augmented[idx] = batch[idx]
    return batch_augmented


def normalization_factors(data, train_idx, shape, mode="slice"):
    """ 
    Shape should be of length 3. 
    mode : either "slice" or "voxel" - defines the granularity of the 
    normalization. Voxelwise normalization does not work well with only
    linear registered data.
    """
    print("Computing the normalization factors of the training data..")
    if mode == "slice":
        axis = (0, 1, 2, 3)
    elif mode == "voxel":
        axis = 0
    else:
        raise NotImplementedError("Normalization mode unknown.")
    print(data.shape)
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    return np.squeeze(mean), np.squeeze(std)

class Normalize(object):
    """
    Normalize tensor with first and second moments.
    By default will only normalize on non-zero voxels. Set 
    masked = False if this is undesired.
    """

    def __init__(self, mean, std=1, masked=True, eps=1e-10):
        self.mean = mean
        self.std = std
        self.masked = masked
        # set epsilon only if using std scaling
        self.eps = eps if np.all(std) != 1 else 0

    def __call__(self, image):
        if self.masked:
            image = self.zero_masked_transform(image)
        else:
            image = self.apply_transform(image)
        return image

    def denormalize(self, image):
        image = image * (self.std + self.eps) + self.mean
        return image

    def apply_transform(self, image):
        return (image - self.mean) / (self.std + self.eps)

    def zero_masked_transform(self, image):
        """ Only apply transform where input is not zero. """
        img_mask = image == 0
        # do transform
        image = self.apply_transform(image)
        image[img_mask] = 0.
        return image

def shuffle_data(X, y):
    """ Shuffle the dataset. """
    shuffled_idx = np.array(range(len(X)))
    np.random.shuffle(shuffled_idx)
    X = X[shuffled_idx]
    y = y[shuffled_idx]
    return X, y

def replace_classifier(model, activation='softmax', units=2):
    """
    Replace the last layer of you model with a new Dense layer.
    Arguments:
        activation: new activation function
        units: number of outputs units, needs to be equal to 
            number of classes. in binary case set to 1.
    """
    from keras.models import Sequential
    from keras.layers import Dense
    
    model_new = Sequential()
    for layer in model.layers[:-1]:
        model_new.add(layer)
    model_new.add(Dense(units=units, activation=activation))
    return model_new

