import os, time, logging
from sys import modules
# read pngs to save as pdf
from PIL import Image
import matplotlib as mpl
mpl.use("Qt5Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import ants
import numpy as np
import multiprocessing
import zipfile

from .helpers import *
from .deepmask import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# logfile = './logs.log'
logfile = os.path.join('/tmp', str(random_case_id())+'.log')
# create a file handler
try:
    os.remove(logfile)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

handler = logging.FileHandler(logfile)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

os.environ[ "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" ] = str(multiprocessing.cpu_count())
os.environ[ "ANTS_RANDOM_SEED" ] = "666"


class noelImageProcessor:
    def __init__(self, id, t1=None, t2=None, output_dir=None, template=None, usen3=False, args=None, model=None, QC=None, preprocess=True):
        super(noelImageProcessor, self).__init__()
        self._id            = id
        self._t1file        = t1
        self._t2file        = t2
        self._outputdir     = output_dir
        self._template      = template
        self._usen3         = usen3
        self._args          = args
        self._model         = model
        self._QC            = QC
        self._dpi           = 300
        self._transform     = 'Affine'
        self._preprocess    = preprocess


    def __load_nifti_file(self):
    	# load nifti data to memory
        logger.info("loading nifti files")
        print("loading nifti files")
        self._mni = self._template

        if self._t1file == None and self._t2file == None:
        	logger.warn("Please load the data first.", "The data is missing")
        	return

        if self._t1file != None and self._t2file != None:
            self._t1 = ants.image_read( self._t1file )
            self._t2 = ants.image_read( self._t2file )
            self._icbm152 = ants.image_read( self._mni )


    def __register_to_MNI_space(self):
        logger.info("registration to MNI template space")
        print("registration to MNI template space")
        if self._t1file != None and self._t2file != None:
            self._t1_reg = ants.registration( fixed = self._icbm152, moving = self._t1, type_of_transform = self._transform )
            self._t2_reg = ants.apply_transforms(fixed = self._icbm152, moving = self._t2, transformlist = self._t1_reg['fwdtransforms'])
            # ants.image_write( self._t1_reg['warpedmovout'], self._t1regfile)
            # ants.image_write( self._t2_reg, self._t2regfile)
            
            
    def __bias_correction(self):
        logger.info("performing {} bias correction".format("N3" if self._usen3 else "N4"))
        print("performing {} bias correction".format("N3" if self._usen3 else "N4"))
        if self._t1file != None and self._t2file != None:
            if self._usen3:
                self._t1_n4 = ants.iMath(ants.n3_bias_field_correction(self._t1_reg['warpedmovout'], downsample_factor=4), "Normalize") * 100
                self._t2_n4 = ants.iMath(ants.n3_bias_field_correction(self._t2_reg, downsample_factor=4), "Normalize") * 100
            else:
                self._t1_n4 = ants.iMath(ants.n4_bias_field_correction(self._t1_reg['warpedmovout']), "Normalize") * 100
                self._t2_n4 = ants.iMath(ants.n4_bias_field_correction(self._t2_reg), "Normalize") * 100
            self._t1regfile = os.path.join(self._outputdir, self._id+'_t1_final.nii.gz')
            self._t2regfile = os.path.join(self._outputdir, self._id+'_t2_final.nii.gz')
            ants.image_write( self._t1_n4, self._t1regfile )
            ants.image_write( self._t2_n4, self._t2regfile )


    def __deepMask_skull_stripping(self):
        logger.info("performing brain extraction using deepMask")
        print("performing brain extraction using deepMask")
        if self._t1file != None and self._t2file != None:
            self._t1brainfile = os.path.join(self._outputdir, self._id+'_t1_brain_final.nii.gz')
            self._t2brainfile = os.path.join(self._outputdir, self._id+'_t2_brain_final.nii.gz')
            if self._preprocess:
                mask = deepMask(self._args, self._model, self._id, self._t1_n4.numpy(), self._t2_n4.numpy(), self._t1regfile, self._t2regfile)
                self._mask = self._t1_n4.new_image_like(mask)
                ants.image_write( self._t1_n4 * self._mask, self._t1brainfile )
                ants.image_write( self._t2_n4 * self._mask, self._t2brainfile )
            else:
                mask = deepMask(self._args, self._model, self._id, self._t1.numpy(), self._t2.numpy(), self._t1file, self._t2file)
                self._mask = self._t1.new_image_like(mask)
                ants.image_write( self._t1 * self._mask, self._t1brainfile )
                ants.image_write( self._t2 * self._mask, self._t2brainfile )


    def __generate_QC_maps(self):
        logger.info('generating QC report')
        if not os.path.exists('./qc'):
            os.makedirs('./qc')
        if self._t1file != None and self._t2file != None:
            self._icbm152.plot(overlay=self._t1, overlay_alpha=0.5, axis=2, ncol=8, nslices=32, title='T1w - Before Registration', filename='./qc/000_t1_before_registration.png', dpi=self._dpi)
            self._icbm152.plot(overlay=self._t1_reg['warpedmovout'], overlay_alpha=0.5, axis=2, ncol=8, nslices=32, title='T1w - After Registration', filename='./qc/001_t1_after_registration.png', dpi=self._dpi)
            self._icbm152.plot(overlay=self._t2, overlay_alpha=0.5, axis=2, ncol=8, nslices=32, title='T2w - Before Registration', filename='./qc/002_t2_before_registration.png', dpi=self._dpi)
            self._icbm152.plot(overlay=self._t2_reg, overlay_alpha=0.5, axis=2, ncol=8, nslices=32, title='T2w - After Registration', filename='./qc/003_t2_after_registration.png', dpi=self._dpi)

            ants.plot(self._t1_reg['warpedmovout'], axis=2, ncol=8, nslices=32, cmap='jet', title='T1w - Before Bias Correction', filename='./qc/004_t1_before_bias_correction.png', dpi=self._dpi)
            ants.plot(self._t1_n4, axis=2, ncol=8, nslices=32, cmap='jet', title='T1w - After Bias Correction', filename='./qc/005_t1_after_bias_correction.png', dpi=self._dpi)
            ants.plot(self._t2_reg, axis=2, ncol=8, nslices=32, cmap='jet', title='T2w - Before Bias Correction', filename='./qc/006_t2_before_bias_correction.png', dpi=self._dpi)
            ants.plot(self._t2_n4, axis=2, ncol=8, nslices=32, cmap='jet', title='T2w - After Bias Correction', filename='./qc/007_t2_after_bias_correction.png', dpi=self._dpi)

            self._t1_n4.plot(overlay=self._mask, overlay_alpha=0.5, axis=2, ncol=8, nslices=32, title='Brain Masking', filename='./qc/008_brain_masking.png', dpi=self._dpi)

            with PdfPages(os.path.join(self._outputdir, self._id+"_QC_report.pdf")) as pdf:
                for i in sorted(os.listdir('./qc')):
                    if i.endswith(".png"):
                        plt.figure()
                        img = Image.open(os.path.join('./qc', i))
                        plt.imshow(img)
                        plt.axis('off')
                        pdf.savefig(dpi=self._dpi)
                        plt.close()
                        os.remove(os.path.join('./qc', i))

    def __create_zip_archive(self):
        print('creating a zip archive')
        logger.info('creating a zip archive')
        zip_archive = zipfile.ZipFile(os.path.join(self._outputdir, self._id + "_archive.zip"), 'w')
        for folder, _, files in os.walk(self._outputdir):
            for file in files:
                if file.endswith('.nii.gz'):
                    zip_archive.write(os.path.join(folder, file), file, compress_type = zipfile.ZIP_DEFLATED)
        zip_archive.close()

    def pipeline(self):
        start = time.time()
        if self._t1file.lower().endswith(('.nii.gz', '.nii')):
            self.__load_nifti_file()
        else:
            raise Exception("Sorry, only NIfTI format is currently supported")
        
        if self._preprocess:
            self.__register_to_MNI_space()
            self.__bias_correction()
            self.__deepMask_skull_stripping()
        else:
            print("Skipping image preprocessing, presumably images are co-registered and bias-corrected")
            self.__deepMask_skull_stripping()

        if self._QC:
            self.__generate_QC_maps()
        # self.__create_zip_archive()
        end = time.time()
        print("pipeline processing time elapsed: {} seconds".format(np.round(end-start, 1)))
        logger.info("pipeline processing time elapsed: {} seconds".format(np.round(end-start, 1)))
        logger.info("*********************************************")