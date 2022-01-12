import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from Classes.Params import param
import tensorflow as tf
tf.random.set_seed(param.SEED) 
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
np.random.seed(param.SEED)   
import gzip
from scipy.ndimage.interpolation import shift
import SimpleITK as sitk
from glob import glob
import random
from dltk.io.augmentation import *
from dltk.io.preprocessing import *




class TFRec():
    
    def __init__(self, fname):
        self.fname = fname
        self.tfwriter = tf.io.TFRecordWriter(self.fname)
        
    def _bytes_feature(self, nparr):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[nparr.tobytes()]))     

    def _float_feature(self, nparr):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=nparr)) 

    def write_record(self, image, mask):

        feature = {
            'image_raw': self._float_feature(image.ravel()),          
            'img_shape': self._bytes_feature(np.array(image.shape, dtype=np.float32).ravel()),
            'mask_raw': self._float_feature(mask.ravel()),
            'mask_shape': self._bytes_feature(np.array(mask.shape, dtype=np.float32).ravel())
        }

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.tfwriter.write(tf_example.SerializeToString())
        
    def close_record(self):
        self.tfwriter.flush()
        self.tfwriter.close()


class TFRsaver():
    def __init__(self, tfr):
        self.qtfr = tfr
        
    def savedata( self, img, mask):     
        self.qtfr.write_record( 
            (img).astype(np.int64),
            (mask).astype(np.int64))


class DataRead():
    def __init__(self):
        self.feature_description = {         
            'image_raw': tf.io.VarLenFeature( dtype=tf.int64),
            'img_shape': tf.io.FixedLenFeature([], tf.string),
            'mask_raw': tf.io.VarLenFeature( dtype=tf.int64),
            'mask_shape': tf.io.FixedLenFeature([], tf.string),         
        }  

    def prepdata( self, fmap):
        pmap = tf.io.parse_single_example(fmap, self.feature_description)

        imgraw = tf.sparse.to_dense(pmap['image_raw'])
        imshape =  tf.io.decode_raw(pmap['img_shape'], tf.float32)
        maskraw = tf.sparse.to_dense(pmap['mask_raw'])
        maskshape =  tf.io.decode_raw(pmap['mask_shape'], tf.float32)
                                    
        return (tf.reshape( imgraw, tf.cast(imshape, tf.int64)),
                tf.reshape( maskraw, tf.cast(maskshape, tf.int64)))


class BRATS():

    def __init__(self, 
                cwd = param.cwd,
                REMOTE = param.REMOTE,
                NUM_CLIENTS = param.NUM_CLIENTS,
                num_classes = param.num_classes,
                SHUFFLE_BUFFER = param.SHUFFLE_BUFFER,
                PREFETCH_BUFFER = param.PREFETCH_BUFFER,
                BATCH_SIZE = param.BATCH_SIZE):

        self.cwd = cwd
        self.REMOTE = REMOTE
        self.NUM_CLIENTS = NUM_CLIENTS
        self.num_classes = num_classes
        self.SHUFFLE_BUFFER = SHUFFLE_BUFFER
        self.PREFETCH_BUFFER = PREFETCH_BUFFER
        self.BATCH_SIZE = BATCH_SIZE

        # Choose rights directory
        if self.REMOTE:

            filename = glob(glob(self.cwd + '/HGG/training/B*')[0] + '/*flair.nii.gz')

        else:

            filename = glob(glob(self.cwd + '/H0/training/B*')[0] + '/*flair.nii.gz')

        # Computing input shape
        example_image = sitk.ReadImage(filename)
        example_image = sitk.GetArrayFromImage(example_image)
        example_image = np.squeeze(example_image)
        input_tensor_shape = example_image.shape
        
        # Preparation of input data
        self.height = input_tensor_shape[1] 
        self.width = input_tensor_shape[2] 
        self.levels = 4                            # FLAIR, T1, T1CE, T2
        self.slices = [0,input_tensor_shape[0]-1]  # slice to consider

        if self.REMOTE:

            self.num_images_train = []
            DIR = (self.cwd + '/HGG/training')
            number_files = len(next(os.walk(DIR))[1])
            self.num_images_train.append((self.slices[1]-self.slices[0])*number_files)
            # print('Number training files: '+ str(number_files))
                
            self.num_images_valid = []
            DIR = (self.cwd + '/HGG/validation')
            number_files = len(next(os.walk(DIR))[1])
            self.num_images_valid.append((self.slices[1]-self.slices[0])*number_files)
            # print('Number training files: '+ str(number_files))     
                      
            self.num_images_test = []
            DIR = (self.cwd + '/HGG/test')
            number_files = len(next(os.walk(DIR))[1])
            self.num_images_test.append((self.slices[1]-self.slices[0])*number_files)
            # print('Number training files: '+ str(number_files))

        else:

            self.num_images_train = []
            for client_ in range(self.NUM_CLIENTS):
                DIR = (self.cwd + '/H{}/training').format(client_)
                number_files = len(next(os.walk(DIR))[1])
                self.num_images_train.append((self.slices[1]-self.slices[0])*number_files)
                # print('Number training files for H{}: '.format(client_) + str(number_files))
                
            self.num_images_valid = []
            for client_ in range(self.NUM_CLIENTS):
                DIR = (self.cwd + '/H{}/validation').format(client_)
                number_files = len(next(os.walk(DIR))[1])
                self.num_images_valid.append((self.slices[1]-self.slices[0])*number_files)
                # print('Number validation samples for H{}: '.format(client_) + str(number_samples))
                
            self.num_images_test = []
            for client_ in range(self.NUM_CLIENTS):
                DIR = (self.cwd + '/H{}/test').format(client_)
                number_files = len(next(os.walk(DIR))[1])
                self.num_images_test.append((self.slices[1]-self.slices[0])*number_files)
                # print('Number testing samples for H{}: '.format(client_) + str(number_samples))

    # Number images
    def return_num_images_train(self):
        return self.num_images_train
    
    def return_num_images_valid(self):
        return self.num_images_valid
    
    def return_num_images_test(self):
        return self.num_images_test
    
    # Input shape
    def return_input_shape(self):
        return [self.height, self.width, self.levels]


    def create_tfrecord(self, client_id = None, mode = None):
        
        self.client_id = client_id
        self.apply_data_augmentation = False
        self.mode = mode

        # Brats dataset levels = [0,1,2,4] -> [0,1,2,3]
        def prepare_target(x_, y_):
            y_[y_ == 4] = 3
            return x_, y_
        
        # Image augmentation function
        def _augment(img):
            
            img = tf.image.random_flip_left_right(img)
            img = elastic_transform(img, alpha=[1e5, 1e5, 1], sigma=[50, 50, 1])
            
            # Separate target from image
            y = img[:,:,4]
            img = img[:,:,0:4]
            img, y = prepare_target(img, y)
            y =  y[..., np.newaxis]
            # y =  np.array([tf.cast(np.any(y),  tf.float32)]) # return 1 if the tumor is present, 0 otherwise
            
            img = tf.image.random_contrast(img, lower=0.0, upper=1.0)
            img = add_gaussian_offset(img, sigma=0.3)
            img = add_gaussian_noise(img, sigma=0.1)    
                                        
            return img,  y
        
        def read_fn(file_references, params=None):

            if self.REMOTE:

                if self.mode == 'train':
                    filename = (self.cwd + '/HGG/training')
                elif self.mode == 'test':
                    filename = (self.cwd + '/HGG/test')
                elif self.mode == 'valid':
                    filename = (self.cwd + '/HGG/validation')
                else:
                    raise ValueError("Invalid Mode")
                    
            else:

                if self.mode == 'train':
                    filename = (self.cwd + '/H{}/training').format(self.client_id)
                elif self.mode == 'test':
                    filename = (self.cwd + '/H{}/test').format(self.client_id)
                elif self.mode == 'valid':
                    filename = (self.cwd + '/H{}/validation').format(self.client_id)
                else:
                    raise ValueError("Invalid Mode")

            # If already exists tfrecord file, nothing to do
            if len(glob(filename + '/tfrec*')) != 0:

                print('tfrecord for {} already existing\n'.format(self.mode))

                return

            else:

                print('Start creating tfrecord\n')

            tfr = TFRec(filename + '/tfrecord')
            tfrsaver = TFRsaver(tfr)

            patients = glob(filename + '/B*')
            random.shuffle(patients)

            for patient in patients:
                
                exam = glob(patient + '/*')
                
                data_path_fl = glob(patient + '/*flair.nii.gz')[0]
                data_path_t1 = glob(patient + '/*t1.nii.gz')[0]
                data_path_t1ce = glob(patient + '/*t1ce.nii.gz')[0]
                data_path_t2 = glob(patient + '/*t2.nii.gz')[0]

                data_path_segm = glob(patient + '/*seg.nii.gz')[0]

                # Read the .nii image containing a brain volume with SimpleITK and get 
                # the numpy array:
                sitk_fl = sitk.ReadImage(data_path_fl)
                image_fl = sitk.GetArrayFromImage(sitk_fl) 

                sitk_t1 = sitk.ReadImage(data_path_t1) 
                image_t1 = sitk.GetArrayFromImage(sitk_t1)

                sitk_t1ce = sitk.ReadImage(data_path_t1ce) 
                image_t1ce = sitk.GetArrayFromImage(sitk_t1ce)

                sitk_t2 = sitk.ReadImage(data_path_t2)
                image_t2 = sitk.GetArrayFromImage(sitk_t2)

                # Read the .nii image containing the segmented image with SimpleITK and get 
                # the numpy array:
                sitk_segm = sitk.ReadImage(data_path_segm)
                image_segm = sitk.GetArrayFromImage(sitk_segm)      

                # Normalise the image to zero mean/unit std dev:
                image_fl = whitening(image_fl)
                image_t1 = whitening(image_t1)
                image_t1ce = whitening(image_t1ce)
                image_t2 = whitening(image_t2)

                # take only important slices
                for slice_ in range(self.slices[0], self.slices[1]):

                    # the forth channel is the segmented image
                    image = np.stack([image_fl, image_t1, image_t1ce, image_t2, image_segm], axis=-1).astype(np.float32)
                    image = np.squeeze(image[slice_, :, :, :])

                    if self.apply_data_augmentation and self.mode == 'train':

                        image, y = _augment(image)

                    else:

                        # Separate target from image
                        y = image[:,:,4]
                        image = image[:,:,0:4]
                        image, y = prepare_target(image, y)
                        y =  y[..., np.newaxis]
                        # y =  np.array([tf.cast(np.any(y),  tf.float32)]) # return 1 if the tumor is present, 0 otherwise                    

                    tfrsaver.savedata(image, y) 

                print('Processed patient: {} of {}'.format(patient, mode))

            tfr.close_record()

            return
                
        read_fn(file_references=None, params=None)

        return 


    def return_dataset(self, client_id = None, mode = None):

        self.mode = mode
        self.client_id = client_id

        # Create tfrecord file
        self.create_tfrecord(self.client_id, self.mode)

        # Create dataReader
        datar = DataRead()

        if self.REMOTE:

            if self.mode == 'train':
                filename = (self.cwd + '/HGG/training')
            elif self.mode == 'test':
                filename = (self.cwd + '/HGG/test')
            elif self.mode == 'valid':
                filename = (self.cwd + '/HGG/validation')
            else:
                raise ValueError("Invalid Mode")
                
        else:

            if self.mode == 'train':
                filename = (self.cwd + '/H{}/training').format(self.client_id)
            elif self.mode == 'test':
                filename = (self.cwd + '/H{}/test').format(self.client_id)
            elif self.mode == 'valid':
                filename = (self.cwd + '/H{}/validation').format(self.client_id)
            else:
                raise ValueError("Invalid Mode")

        # If tfrecord file does not exist, throws exception
        if len(glob(filename + '/tf*')) == 0:

            raise ValueError("Dataset does not exist, Create it with create_dataset()")

        # Recover dataset and conver to tf.data.TFRecordDataset
        tfrds = tf.data.TFRecordDataset(glob(filename + '/tfrec*')[0])
        dataset = tfrds.map(datar.prepdata, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.cache()    # only if we have enough memory

        return dataset


    # Preprocess using labels (no OHE)
    # def preprocess(self, dataset):
        
    #     def batch_format_fn(image, label):

    #         return tf.reshape(image, [-1, self.height, self.width, self.levels]), tf.reshape(label, [-1,  self.height, self.width, 1])
        
    #     return dataset.repeat().shuffle(self.SHUFFLE_BUFFER).batch(self.BATCH_SIZE).map(batch_format_fn).prefetch(self.PREFETCH_BUFFER)


    # Preprocess (OHE)
    def preprocess(self, dataset):

        def to_categorical(x_, y_):
        
            return x_, tf.one_hot(tf.cast(y_, tf.uint8), depth=self.num_classes)
        
        def batch_format_fn(image, label):

            return tf.reshape(image, [-1, self.height, self.width, self.levels]), tf.reshape(label, [-1,  self.height, self.width, self.num_classes])
        
        return dataset.repeat().shuffle(self.SHUFFLE_BUFFER).batch(self.BATCH_SIZE).map(to_categorical).map(batch_format_fn).prefetch(self.PREFETCH_BUFFER)

    def create_tf_dataset_for_client_train(self, client_id = None):
        return  self.preprocess(self.return_dataset(client_id, mode = 'train'))

    def create_tf_dataset_for_client_valid(self, client_id = None):
        return  self.preprocess(self.return_dataset(client_id, mode = 'valid'))

    def create_tf_dataset_for_client_test(self, client_id = None):
        return  self.preprocess(self.return_dataset(client_id, mode = 'test'))


    # def create_tf_dataset_train(self):
    #     return  self.preprocess(return_dataset(mode = 'train'))

    # def create_tf_dataset_valid(self):
    #     return  self.preprocess(return_dataset(mode = 'valid'))

    # def create_tf_dataset_test(self):
    #     return  self.preprocess(return_dataset(mode = 'test'))

