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
import operator
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import re
import difflib
from pydicom import dcmread
from pydicom.data import get_testdata_file
import cv2

def whitening_slice(image):
    """Whitening. Normalises slices to zero mean and unit variance."""
    image = image.astype(np.float32)
    for slice in range(image.shape[0]):
        image[slice] = whitening(image[slice])
    return image

class TFRec():
    
    def __init__(self, fname):
        self.fname = fname
        self.tfwriter = tf.io.TFRecordWriter(self.fname)
        
    def _bytes_feature(self, nparr):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[nparr.tobytes()]))     

    def _float_feature(self, nparr):
        return tf.train.Feature(float_list=tf.train.FloatList(value=nparr)) 

    def write_record(self, image, mask):

        feature = {
            'image_raw': self._float_feature(image.ravel()),          
            'img_shape': self._bytes_feature(np.array(image.shape, dtype= np.float32).ravel()),
            'mask_raw': self._float_feature(mask.ravel()),
            'mask_shape': self._bytes_feature(np.array(mask.shape, dtype= np.float32).ravel())
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
            (img).astype(np.float32),
            (mask).astype(np.float32))


class DataRead():
    def __init__(self):
        self.feature_description = {         
            'image_raw': tf.io.VarLenFeature( dtype=tf.float32),
            'img_shape': tf.io.FixedLenFeature([], tf.string),
            'mask_raw': tf.io.VarLenFeature( dtype=tf.float32),
            'mask_shape': tf.io.FixedLenFeature([], tf.string),         
        }  

    def prepdata( self, fmap):
        pmap = tf.io.parse_single_example(fmap, self.feature_description)

        imgraw = tf.sparse.to_dense(pmap['image_raw'])
        imshape =  tf.io.decode_raw(pmap['img_shape'],  tf.float32)
        maskraw = tf.sparse.to_dense(pmap['mask_raw'])
        maskshape =  tf.io.decode_raw(pmap['mask_shape'],  tf.float32)
                                    
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
                BATCH_SIZE = param.BATCH_SIZE,
                number_medical_exams = param.number_medical_exams):

        self.cwd = cwd
        self.REMOTE = REMOTE
        self.NUM_CLIENTS = NUM_CLIENTS
        self.num_classes = num_classes
        self.SHUFFLE_BUFFER = SHUFFLE_BUFFER
        self.PREFETCH_BUFFER = PREFETCH_BUFFER
        self.BATCH_SIZE = BATCH_SIZE
        self.number_medical_exams = number_medical_exams

        # Dataset organization
        # NOT REMOTE case:
        # - Server/Client:
        #     - H0:
        #         - training:
        #             - Patient1
        #             - Patient2
        #             ...
        #             - num_images_train
        #         - test:
        #             ...    
        #         - validation:
        #             ...              
        #     ...
        #     - HN:  
        #         ...

        # REMOTE case:
        # - Server:
        #     - H0:
        #         - training:
        #             - num_images_train
        #         - test:
        #             - num_images_test    
        #         - validation:
        #             - num_images_valid             
        #     ...
        #     - HN:  
        #         ...  
        # - Client_i:
        #     - H0:
        #         - training:
        #             - num_images_train
        #         - test:
        #             - num_images_test    
        #         - validation:
        #             - num_images_valid     
        #     - Hi:
        #         - training:
        #             - Patient1
        #             - Patient2
        #             ...
        #             - num_images_train
        #         - test:
        #             - Patient1
        #             - Patient2
        #             ...
        #             - num_images_test    
        #         - validation:
        #             - Patient1
        #             - Patient2
        #             ...
        #             - num_images_valid          
        #     ...
        #     - HN:  
        #         - training:
        #             - num_images_train
        #         - test:
        #             - num_images_test    
        #         - validation:
        #             - num_images_valid     

        # Choose rights directory 
        dataset_directories = glob(os.path.join(self.cwd, 'H*'))
        
        # Preparation of input data
        self.height = 240                       # input_tensor_shape[1] 
        self.width = 240                        # input_tensor_shape[2] 
        self.levels = self.number_medical_exams # FLAIR, T1, T1CE, T2
        self.slices = [0,155]  # slice to consider

        # Recover Client id
        self.CLIENTS = []
        for elem in next(os.walk(self.cwd))[1]:
            if 'H' in elem:
                try:
                    id_ = int(param.re.findall('\d+',elem)[0])
                    self.CLIENTS.append(id_)
                except: 
                    pass

        # Remote training
        if self.REMOTE:

            self.num_images_train = {}
            for client_ in self.CLIENTS:
                DIR = os.path.join(self.cwd, 'H{}'.format(client_), 'training')
                if os.path.exists(DIR):
                    if not os.path.exists(os.path.join(DIR, 'number_training_files.npy')):
                        print('Creating number_training_files.npy')
                        if len(next(os.walk(DIR))[1]) == 0:
                            print("Warning, zero files in training folder\n")
                            np.save(os.path.join(DIR, 'number_training_files.npy'), 0, allow_pickle = True)
                        else:
                            patients = next(os.walk(DIR))[1]
                            number_patients = len(patients)

                            num_slices = 0
                            for patient in patients:
                                # Verify format 
                                format = 'nii' if 'nii' in glob(os.path.join(DIR, patient, '*'))[0] else 'dcm'

                                # Count slices 
                                if format == 'nii':                   
                                    num_slices += (self.slices[1]-self.slices[0])
                                elif format == 'dcm':
                                    num_slices += len(glob(os.path.join(DIR, patient, '*.dcm')))

                            self.num_images_train[client_] = num_slices
                            print('Number training files for H{}: '.format(client_) + str(self.num_images_train[client_]))

                            # Save number of files
                            np.save(os.path.join(DIR, 'number_training_files.npy'), self.num_images_train[client_], allow_pickle = True)

            self.num_images_valid = {}
            for client_ in self.CLIENTS:
                DIR = os.path.join(self.cwd, 'H{}'.format(client_), 'validation')
                if os.path.exists(DIR):
                    if not os.path.exists(os.path.join(DIR, 'number_validation_files.npy')):
                        print('Creating number_validation_files.npy')
                        if len(next(os.walk(DIR))[1]) == 0:
                            print("Warning, zero files in validation folder\n")
                            np.save(os.path.join(DIR, 'number_validation_files.npy'), 0, allow_pickle = True)
                        else:
                            patients = next(os.walk(DIR))[1]
                            number_patients = len(patients)

                            num_slices = 0
                            for patient in patients:
                                # Verify format
                                format = 'nii' if 'nii' in glob(os.path.join(DIR, patient, '*'))[0] else 'dcm' 

                                # Count slices 
                                if format == 'nii':                   
                                    num_slices += (self.slices[1]-self.slices[0])
                                elif format == 'dcm':
                                    num_slices += len(glob(os.path.join(DIR, patient, '*.dcm')))

                            self.num_images_valid[client_] = num_slices
                            print('Number validation files for H{}: '.format(client_) + str(self.num_images_valid[client_]))

                            # Save number of files
                            np.save(os.path.join(DIR, 'number_validation_files.npy'), self.num_images_valid[client_], allow_pickle = True)

            self.num_images_test = {}
            for client_ in self.CLIENTS:
                DIR = os.path.join(self.cwd, 'H{}'.format(client_), 'test')
                if os.path.exists(DIR):
                    if not os.path.exists(os.path.join(DIR, 'number_testing_files.npy')):                    
                        print('Creating number_test_files.npy')
                        if len(next(os.walk(DIR))[1]) == 0:
                            print("Warning, zero files in test folder\n")
                            np.save(os.path.join(DIR, 'number_testing_files.npy'), 0, allow_pickle = True)
                        else:
                            patients = next(os.walk(DIR))[1]
                            number_patients = len(patients)

                            num_slices = 0
                            for patient in patients:
                                # Verify format
                                format = 'nii' if 'nii' in glob(os.path.join(DIR, patient, '*'))[0] else 'dcm' 

                                # Count slices 
                                if format == 'nii':                   
                                    num_slices += (self.slices[1]-self.slices[0])
                                elif format == 'dcm':
                                    num_slices += len(glob(os.path.join(DIR, patient, '*.dcm')))

                            self.num_images_test[client_] = num_slices
                            print('Number testing files for H{}: '.format(client_) + str(self.num_images_test[client_]))

                            # Save number of files
                            np.save(os.path.join(DIR, 'number_testing_files.npy'), self.num_images_test[client_], allow_pickle = True)

        # Not Remote training
        else:

            self.num_images_train = {}
            for client_ in range(self.NUM_CLIENTS):
                DIR = os.path.join(self.cwd, 'H{}'.format(client_), 'training')
                patients = next(os.walk(DIR))[1]
                number_patients = len(patients)

                num_slices = 0
                for patient in patients:
                    # Verify format
                    format = 'nii' if 'nii' in glob(os.path.join(DIR, patient, '*'))[0] else 'dcm' 

                    # Count slices 
                    if format == 'nii':                   
                        num_slices += (self.slices[1]-self.slices[0])
                    elif format == 'dcm':
                        num_slices += len(glob(os.path.join(DIR, patient, '*.dcm')))

                self.num_images_train[client_] = num_slices
                print('Number training files for H{}: '.format(client_) + str(self.num_images_train[client_]))

                # Save number of files
                np.save(os.path.join(DIR, 'number_training_files.npy'), self.num_images_train[client_], allow_pickle = True)

            self.num_images_valid = {}
            for client_ in range(self.NUM_CLIENTS):
                DIR = os.path.join(self.cwd, 'H{}'.format(client_), 'validation')
                patients = next(os.walk(DIR))[1]
                number_patients = len(patients)

                num_slices = 0
                for patient in patients:
                    # Verify format
                    format = 'nii' if 'nii' in glob(os.path.join(DIR, patient, '*'))[0] else 'dcm' 

                    # Count slices 
                    if format == 'nii':                   
                        num_slices += (self.slices[1]-self.slices[0])
                    elif format == 'dcm':
                        num_slices += len(glob(os.path.join(DIR, patient, '*.dcm')))

                self.num_images_valid[client_] = num_slices
                print('Number validation files for H{}: '.format(client_) + str(self.num_images_valid[client_]))

                # Save number of files
                np.save(os.path.join(DIR, 'number_validation_files.npy'), self.num_images_valid[client_], allow_pickle = True)

            self.num_images_test = {}
            for client_ in range(self.NUM_CLIENTS):
                DIR = os.path.join(self.cwd, 'H{}'.format(client_), 'test')
                patients = next(os.walk(DIR))[1]
                number_patients = len(patients)

                num_slices = 0
                for patient in patients:
                    # Verify format
                    format = 'nii' if 'nii' in glob(os.path.join(DIR, patient, '*'))[0] else 'dcm' 

                    # Count slices 
                    if format == 'nii':                   
                        num_slices += (self.slices[1]-self.slices[0])
                    elif format == 'dcm':
                        num_slices += len(glob(os.path.join(DIR, patient, '*.dcm')))

                self.num_images_test[client_] = num_slices
                print('Number testing files for H{}: '.format(client_) + str(self.num_images_test[client_]))

                # Save number of files
                np.save(os.path.join(DIR, 'number_testing_files.npy'), self.num_images_test[client_], allow_pickle = True)

    # Number images
    def return_num_images_train(self):
        self.num_images_train = {}
        for client_ in self.CLIENTS:
            DIR = os.path.join(self.cwd, 'H{}'.format(client_), 'training')
            if os.path.exists(DIR):
                self.num_images_train[client_] = (np.load(os.path.join(DIR, 'number_training_files.npy'), allow_pickle = True))
            else:
                self.num_images_train[client_] = 0
        return self.num_images_train
    
    def return_num_images_valid(self):
        self.num_images_valid = {}
        for client_ in self.CLIENTS:
            DIR = os.path.join(self.cwd, 'H{}'.format(client_), 'validation')
            if os.path.exists(DIR):
                self.num_images_valid[client_] = (np.load(os.path.join(DIR, 'number_validation_files.npy'), allow_pickle = True))
            else:
                self.num_images_valid[client_] = 0
        return self.num_images_valid
    
    def return_num_images_test(self):
        self.num_images_test = {}
        for client_ in self.CLIENTS:
            DIR = os.path.join(self.cwd, 'H{}'.format(client_), 'test')
            if os.path.exists(DIR):
                self.num_images_test[client_] = (np.load(os.path.join(DIR, 'number_testing_files.npy'), allow_pickle = True))
            else:
                self.num_images_test[client_] = 0
        return self.num_images_test
    
    # Input shape
    def return_input_shape(self):
        return [self.height, self.width, self.levels]


    def create_tfrecord(self, client_id = None, mode = None):
        
        self.client_id = client_id
        self.apply_data_augmentation = 1
        self.mode = mode

        # Brats dataset levels = [0,1,2,4] -> [0,1,2,3]
        def prepare_target(x_, y_):
            y_[y_ == 4] = 3
            return x_, y_
        
        def cropND(img, bounding):
            start = tuple(map(lambda a, da: a/2-da/2, img.shape, bounding))
            end = tuple(map(operator.add, start, bounding))
            slices = tuple(map(slice, start, end))
            return img[slices]
        
        # Image augmentation function
        def _augment(img):

            if np.random.rand() > 0.5: 
                # tf.image.random_flip_left_right(img)
                ax = np.random.choice([0,1])
                img = np.flip(img, ax)
            if np.random.rand() > 0.5:
                # img = elastic_transform(img, alpha=[1e5, 1e5, 1], sigma=[50, 50, 1])
                pass
            if np.random.rand() > 0.5:
                rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
                img = np.rot90(img, rot, axes=[0,1])  # Rotate axes 0 and 1

            # Separate target from image
            y = img[:,:,self.levels]
            img = img[:,:,0:self.levels]
            img, y = prepare_target(img, y)
            y =  y[..., np.newaxis]
            # y =  np.array([tf.cast(np.any(y),  tf.float32)]) # return 1 if the tumor is present, 0 otherwise
            
            # img = tf.image.random_contrast(img, lower=0.0, upper=1.0)
            if np.random.rand() > 0.5:
                # img = add_gaussian_offset(img, sigma=0.3)
                img = add_gaussian_noise(img, sigma=0.1)    
                                        
            return img,  y
        
        def read_fn(file_references, params=None):

            if self.mode == 'train':
                filename = os.path.join(self.cwd, 'H{}'.format(self.client_id), 'training')
            elif self.mode == 'test':
                filename = os.path.join(self.cwd, 'H{}'.format(self.client_id), 'test')
            elif self.mode == 'valid':
                filename = os.path.join(self.cwd, 'H{}'.format(self.client_id), 'validation')
            else:
                raise ValueError("Invalid Mode")

            # If already exists tfrecord file, nothing to do
            if len(glob(os.path.join(filename, 'tfrec*'))) != 0:

                print('tfrecord for {} already existing\n'.format(self.mode))

                return

            else:

                print('Start creating tfrecord\n')

            tfr = TFRec(os.path.join(filename, 'tfrecord'))
            tfrsaver = TFRsaver(tfr)

            patients = [os.path.join(next(os.walk(filename))[0], s) for s in next(os.walk(filename))[1]]
            random.shuffle(patients)

            for patient in patients:
                
                exam = glob(os.path.join(patient, '*'))
                
                # Verify format
                format = 'nii' if 'nii' in exam[0] else 'dcm'  

                # If NIfTI format
                if format == 'nii':

                    # Only flair exam
                    if self.levels == 1:
                        data_path_fl = glob(os.path.join(patient, '*flair.nii.gz'))[0]

                        # Read the .nii image containing a brain volume with SimpleITK and get 
                        # the numpy array:
                        sitk_fl = sitk.ReadImage(data_path_fl)
                        image_fl = sitk.GetArrayFromImage(sitk_fl) 

                        # Normalise the image to zero mean/unit std dev:
                        # image_fl = whitening(image_fl)
                        image_fl = whitening_slice(image_fl)
                        

                    # All exam
                    elif self.levels == 4:
                        data_path_fl = glob(os.path.join(patient, '*flair.nii.gz'))[0]
                        data_path_t1 = glob(os.path.join(patient, '*t1.nii.gz'))[0]
                        data_path_t1ce = glob(os.path.join(patient, '*t1ce.nii.gz'))[0]
                        data_path_t2 = glob(os.path.join(patient, '*t2.nii.gz'))[0]

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

                        # Normalise the image to zero mean/unit std dev:
                        # image_fl = whitening(image_fl)
                        # image_t1 = whitening(image_t1)
                        # image_t1ce = whitening(image_t1ce)
                        # image_t2 = whitening(image_t2)
                        image_fl = whitening_slice(image_fl)
                        image_t1 = whitening_slice(image_t1)
                        image_t1ce = whitening_slice(image_t1ce)
                        image_t2 = whitening_slice(image_t2)

                    # Segmentation
                    data_path_segm = glob(os.path.join(patient, '*seg.nii.gz'))[0]
                    
                    # Read the .nii image containing the segmented image with SimpleITK and get 
                    # the numpy array:
                    sitk_segm = sitk.ReadImage(data_path_segm)
                    image_segm = sitk.GetArrayFromImage(sitk_segm) 

                    # Only flair exam
                    if self.levels == 1:
                        # The second channel is the segmented image
                        image_stack = np.stack([image_fl, image_segm], axis=-1).astype(np.float32)     
                    # All exam
                    elif self.levels == 4:
                        # The forth channel is the segmented image
                        image_stack = np.stack([image_fl, image_t1, image_t1ce, image_t2, image_segm], axis=-1).astype(np.float32)

                    # Take only important slices
                    for slice_ in range(self.slices[0], self.slices[1]):

                        image = np.squeeze(image_stack[slice_, :, :, :])

                        if self.apply_data_augmentation and self.mode == 'train':

                            image, y = _augment(image)

                        else:

                            # Separate target from image
                            y = image[:,:,self.levels]
                            image = image[:,:,0:self.levels]
                            image, y = prepare_target(image, y)
                            y =  y[..., np.newaxis]
                            # y =  np.array([tf.cast(np.any(y),  tf.float32)]) # return 1 if the tumor is present, 0 otherwise                    

                        tfrsaver.savedata(image, y) 

                # If DICOM format
                elif format == 'dcm':

                    # Only flair exam
                    if self.levels == 1:
                        data_path_slices_fl = glob(os.path.join(patient, '*Flair*.dcm')) if len(glob(os.path.join(patient, '*Flair*.dcm')))!=0 else glob(os.path.join(patient, '*FLAIR*.dcm'))
                        data_path_segmentations = glob(os.path.join(patient, '*Flair*.png')) if len(glob(os.path.join(patient, '*Flair*.png')))!=0 else glob(os.path.join(patient, '*FLAIR*.png'))

                        for data_path_slice_fl in data_path_slices_fl:

                            # Find corresponding segmentation
                            try: 
                                data_path_segm = difflib.get_close_matches(data_path_slice_fl, data_path_segmentations)[0]
                            except:
                                    try:
                                        data_path_segm = [s for s in data_path_segmentations if data_path_slice_fl.split(os.sep)[-1] in s][0]
                                    except:
                                        raise ValueError("Enabled to find correspondence exam-segmentation in patient: {}, slice: {}".format(patient, data_path_slice_fl))

                            # Read the .dcm image containing a brain slice with pydicom and get 
                            # the numpy array:
                            pydicom_fl = dcmread(data_path_slice_fl)
                            image_fl = pydicom_fl.pixel_array

                            # Resize image
                            image_fl = cv2.resize(image_fl, dsize=(self.height, self.width), interpolation=cv2.INTER_CUBIC)
                            # image_fl = cropND(image_fl, (self.height, self.width))

                            # Normalise the image to zero mean/unit std dev:
                            image_fl = whitening(image_fl)

                            if param.num_classes == 2:
                                # Read the .png image containing the segmented image with pydicom and get 
                                # the numpy array:
                                image_segm = cv2.imread(data_path_segm, cv2.IMREAD_GRAYSCALE)
                                # Resize image
                                image_segm = cv2.resize(image_segm, dsize=(self.height, self.width), interpolation=cv2.INTER_CUBIC)
                                # image_segm = cropND(image_segm, (self.height, self.width))
                                image_segm = cv2.threshold(image_segm, 127, 255, cv2.THRESH_BINARY)[1]
                            else:
                                raise ValueError("To be implemented, multiclass tumor classification with DICOM format")

                            # The second channel is the segmented image
                            image = np.stack([image_fl, image_segm], axis=-1).astype(np.float32) 

                            if self.apply_data_augmentation and self.mode == 'train':
        
                                image, y = _augment(image)

                            else:

                                # Separate target from image
                                y = image[:,:,self.levels]
                                image = image[:,:,0:self.levels]
                                image, y = prepare_target(image, y)
                                y =  y[..., np.newaxis]
                                # y =  np.array([tf.cast(np.any(y),  tf.float32)]) # return 1 if the tumor is present, 0 otherwise                    

                            tfrsaver.savedata(image, y) 

                    # All exam
                    elif self.levels == 4:
                        raise ValueError("Other exams to be implemented")


                # # If DICOM format
                # elif format == 'dcm':
                    
                #     # Only flair exam
                #     if self.levels == 1:
                #         data_path_slices_fl = glob(os.path.join(patient, '*Flair*.dcm')) if len(glob(os.path.join(patient, '*Flair*.dcm')))!=0 else glob(os.path.join(patient, '*FLAIR*.dcm'))
                #         data_path_segmentations = glob(os.path.join(patient, '*Flair*.png')) if len(glob(os.path.join(patient, '*Flair*.png')))!=0 else glob(os.path.join(patient, '*FLAIR*.png'))

                #         image_fl = []
                #         image_segm = []

                #         for data_path_slice_fl in data_path_slices_fl:

                #             # Find corresponding segmentation
                #             try: 
                #                 data_path_segm = difflib.get_close_matches(data_path_slice_fl, data_path_segmentations)[0]
                #             except:
                #                     try:
                #                         data_path_segm = [s for s in data_path_segmentations if data_path_slice_fl.split(os.sep)[-1] in s][0]
                #                     except:
                #                         raise ValueError("Enabled to find correspondence exam-segmentation in patient: {}, slice: {}".format(patient, data_path_slice_fl))

                #             # Read the .dcm image containing a brain slice with pydicom and get 
                #             # the numpy array:
                #             pydicom_fl = dcmread(data_path_slice_fl)
                #             slice_fl = pydicom_fl.pixel_array

                #             # Resize image
                #             slice_fl = cv2.resize(slice_fl, dsize=(self.height, self.width), interpolation=cv2.INTER_CUBIC)
                #             # image_fl = cropND(image_fl, (self.height, self.width))

                #             if param.num_classes == 2:
                #                 # Read the .png image containing the segmented image with pydicom and get 
                #                 # the numpy array:
                #                 slice_segm = cv2.imread(data_path_segm, cv2.IMREAD_GRAYSCALE)
                #                 # Resize image
                #                 slice_segm = cv2.resize(slice_segm, dsize=(self.height, self.width), interpolation=cv2.INTER_CUBIC)
                #                 # image_segm = cropND(image_segm, (self.height, self.width))
                #                 slice_segm = cv2.threshold(slice_segm, 127, 255, cv2.THRESH_BINARY)[1]
                #             else:
                #                 raise ValueError("To be implemented, multiclass tumor classification with DICOM format")

                #             image_fl.append(slice_fl)
                #             image_segm.append(slice_segm)

                #         # create 3D images
                #         image_fl = np.array(image_fl)
                #         image_segm = np.array(image_segm)

                #         # Normalise the image to zero mean/unit std dev:
                #         image_fl = whitening(image_fl)

                #         # The second channel is the segmented image
                #         image_stack = np.stack([image_fl, image_segm], axis=-1).astype(np.float32) 


                #         for slice_ in range(0, image_fl.shape[0]):

                #             image = np.squeeze(image_stack[slice_, :, :, :])

                #             if self.apply_data_augmentation and self.mode == 'train':

                #                 image, y = _augment(image)

                #             else:

                #                 # Separate target from image
                #                 y = image[:,:,self.levels]
                #                 image = image[:,:,0:self.levels]
                #                 image, y = prepare_target(image, y)
                #                 y =  y[..., np.newaxis]
                #                 # y =  np.array([tf.cast(np.any(y),  tf.float32)]) # return 1 if the tumor is present, 0 otherwise                    

                #             tfrsaver.savedata(image, y) 

                    # # All exam
                    # elif self.levels == 4:
                    #     raise ValueError("Other exams to be implemented")

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

        if self.mode == 'train':
            filename = os.path.join(self.cwd, 'H{}'.format(self.client_id), 'training')
        elif self.mode == 'test':
            filename = os.path.join(self.cwd, 'H{}'.format(self.client_id), 'test')
        elif self.mode == 'valid':
            filename = os.path.join(self.cwd, 'H{}'.format(self.client_id), 'validation')
            
        else:
            raise ValueError("Invalid Mode")

        # If tfrecord file does not exist, throws exception
        if len(glob(os.path.join(filename, 'tf*'))) == 0:

            raise ValueError("Dataset does not exist, Create it with create_dataset()")

        # Recover dataset and conver to tf.data.TFRecordDataset
        tfrds = tf.data.TFRecordDataset(glob(os.path.join(filename, 'tfrec*'))[0])
        dataset = tfrds.map(datar.prepdata, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.mode == 'train' and param.CACHE_DATASET_TRAIN:
            dataset = dataset.cache()    # only if we have enough memory
        if self.mode == 'valid' and param.CACHE_DATASET_VALID:
            dataset = dataset.cache()    # only if we have enough memory
        return dataset


    # Preprocess using labels (no OHE)
    # def preprocess(self, dataset):
        
    #     def batch_format_fn(image, label):

    #         return tf.reshape(image, [-1, self.height, self.width, self.levels]), tf.reshape(label, [-1,  self.height, self.width, 1])
        
    #     return dataset.repeat().shuffle(self.SHUFFLE_BUFFER).batch(self.BATCH_SIZE).map(batch_format_fn).prefetch(self.PREFETCH_BUFFER)


    # Preprocess (OHE)
    def preprocess(self, dataset):

        def set_channels(x_, y_):
            
            if self.number_medical_exams == 1:
                x_ = x_[:,:,0:self.number_medical_exams]
            return x_,y_

        def to_categorical(x_, y_):
            
            if self.num_classes == 2:
                y_ = tf.where(tf.greater_equal(y_, 1.0), tf.constant([1.0], dtype=tf.float32), y_)
            return x_, tf.one_hot(tf.cast(y_, tf.uint8), depth=self.num_classes)
        
        def batch_format_fn(image, label):

            return tf.reshape(image, [-1, self.height, self.width, self.levels]), tf.reshape(label, [-1,  self.height, self.width, self.num_classes])
        
        return dataset.repeat().shuffle(self.SHUFFLE_BUFFER).map(set_channels).batch(self.BATCH_SIZE).map(to_categorical).map(batch_format_fn).prefetch(self.PREFETCH_BUFFER)

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

