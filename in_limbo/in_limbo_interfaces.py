import nipy.modalities.fmri.design_matrix as dm
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
import nibabel as nb
from nipype.utils import filemanip
import os
from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, OutputMultiPath, isdefined


import scipy as sp
from scipy import stats
import numpy as np




class GetInLimboWeightsInputSpec(BaseInterfaceInputSpec):
    varcope = traits.File(desc='Variance of contrast')
    contrast = traits.File(desc='Vector of contrast')
    comp_voxel_covariance = traits.File(desc='Covariance with comparison voxel')
    comp_voxel = traits.Tuple(desc='Comparison voxel to make covariance matrix against')
    

class GetInLimboWeightsSandwichOutputSpec(TraitedSpec):
    weights = traits.File(exists=True, desc="Weights for level 2 WLS")


class GetInLimboWeights(BaseInterface):
    input_spec = GetInLimboWeightsInputSpec
    output_spec = GetInLimboWeightsSandwichOutputSpec

    def _run_interface(self, runtime):
        
        varcope_image = nb.load(self.inputs.varcope)
        
        varcope = varcope_image.get_data()        
        covar = nb.load(self.inputs.comp_voxel_covariance).get_data()
        
        print self.inputs.contrast
        contrast = np.loadtxt(self.inputs.contrast)[np.newaxis, :]
        comp_voxel = self.inputs.comp_voxel
        
        V1 = varcope
        V2 = varcope[comp_voxel]
        V3 = np.tensordot(np.tensordot(contrast, covar, (1,1)), contrast, (1, 1)).squeeze()

        W = 1 / np.sqrt(V1 + V2 - 2*V3)
        
        # Against rounding errors
        W[np.isnan(W)] = W[~np.isnan(W)].min()

        nb.save(nb.Nifti1Image(W, varcope_image.get_affine()), 'weights.nii.gz')
       
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        
        outputs['weights'] = os.path.abspath('weights.nii.gz')

        return outputs



class GetInLimboAreaInputSpec(BaseInterfaceInputSpec):
    thresholded_z = traits.File(desc='Thresholded z-map')
    in_limbo_t = traits.File(desc='In limbo t-map')
    threshold = traits.Float(desc='(positive) t-threshold for t-map')
    

class GetInLimboAreaOutputSpec(TraitedSpec):
    in_limbo = traits.File(exists=True, desc="Map of area that is in limbo")
    unactivated = traits.File(exists=True, desc="Map of area that is nor in limbo nor activated")    


class GetInLimboArea(BaseInterface):
    input_spec = GetInLimboAreaInputSpec
    output_spec = GetInLimboAreaOutputSpec

    def _run_interface(self, runtime):
        
        
        z_image = nb.load(self.inputs.thresholded_z)
        z = z_image.get_data()
        in_limbo_t = nb.load(self.inputs.in_limbo_t).get_data()

        in_limbo = np.zeros_like(z)
        unactivated = np.zeros_like(z)

        in_limbo[(z == 0) & (in_limbo_t < self.inputs.threshold) & (in_limbo_t != 0)] = 1
        unactivated[(z == 0) & (in_limbo_t > self.inputs.threshold) & (in_limbo_t != 0)] = 1

        nb.save(nb.Nifti1Image(in_limbo, z_image.get_affine(),), 
                os.path.abspath('in_limbo.nii.gz'))

        nb.save(nb.Nifti1Image(unactivated, z_image.get_affine(),), 
            os.path.abspath('in_limbo_unactivated.nii.gz'))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        
        outputs['in_limbo'] = os.path.abspath('in_limbo.nii.gz')
        outputs['unactivated'] = os.path.abspath('in_limbo_unactivated.nii.gz')                                             

        return outputs
