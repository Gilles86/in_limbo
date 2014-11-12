import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.algorithms.modelgen as model
import nipype.interfaces.io as nio
from nipype.interfaces.base import Bunch

from nipype.workflows.fmri.fsl import create_fixed_effects_flow

from .. import OnsetsCutter, OLSFitter, SandwichEstimator, SandwichContrast, GetVoxelCovariance, GetInLimboWeights, Level2WLS, GetInLimboArea

import pandas

import re, glob, os

def create_level1_sandwich(name='level1_sandwich', 
                           base_dir='~/workflow_folders'):

    """Create a level 1 analysis workflow with sandwich covariance
    estimation
    This 
    """
    base_dir = os.path.expanduser(base_dir)
    workflow = pe.Workflow(name=name, base_dir=os.path.abspath(base_dir))


    inputspec = pe.Node(util.IdentityInterface(fields=['functional_runs',
                                                       'session_info',
                                                       'contrasts', 'TR', 'q']),
                        name='inputspec')

    onsets_cutter = pe.MapNode(OnsetsCutter(),
                               iterfield=['session_info'],
                               name='onsets_cutter')
    workflow.connect(inputspec, 'session_info', onsets_cutter, 'session_info')
    workflow.connect(inputspec, 'TR', onsets_cutter, 'TR')
    workflow.connect(inputspec, 'q', onsets_cutter, 'q')

    ols_esimator = pe.MapNode(OLSFitter(), 
                              iterfield=['X', 'data'],
                              name='ols_esimator')

    workflow.connect(onsets_cutter, 'X', ols_esimator, 'X')
    workflow.connect(inputspec, 'functional_runs', ols_esimator, 'data')

    sandwich_estimator = pe.MapNode(SandwichEstimator(), 
                                    iterfield=['beta', 'design_matrices', 'cutted_data'],
                                    name='sandwich_estimator')
    workflow.connect(ols_esimator, 'beta', sandwich_estimator, 'beta')
    workflow.connect(onsets_cutter, 'design_matrices', sandwich_estimator, 'design_matrices')
    workflow.connect(onsets_cutter, 'cutted_data', sandwich_estimator, 'cutted_data')


    contraster = pe.MapNode(SandwichContrast(), 
                         iterfield=['beta', 'variance', 'q', 'session_info'],
                         name='contraster')
    workflow.connect(inputspec, 'contrasts', contraster, 'contrasts')
    workflow.connect(ols_esimator, 'beta', contraster, 'beta')
    workflow.connect(sandwich_estimator, 'variance', contraster, 'variance')
    workflow.connect(onsets_cutter, 'q', contraster, 'q')
    workflow.connect(inputspec, 'session_info', contraster, 'session_info')


    outputspec = pe.Node(util.IdentityInterface(fields=['beta_ols',
                                                        'beta_variance',
                                                        'residuals',
                                                        'design_matrices',
                                                        'dof', 'z_stat',
                                                        't_stat', 'cope',
                                                        'varcope', 'dof',],),
                         name='outputspec')

    workflow.connect(ols_esimator, 'beta', outputspec, 'beta_ols')
    workflow.connect(ols_esimator, 'variance', outputspec, 'beta_variance')
    workflow.connect(sandwich_estimator, 'residuals', outputspec, 'residuals')
    workflow.connect(onsets_cutter, 'design_matrices', outputspec, 'design_matrices')


    # Take sqrt to get standard errors
    sqrt = pe.MapNode(fsl.maths.UnaryMaths(), name='sqrt',
                      iterfield=['in_file'])

    sqrt.inputs.operation  = 'sqrt'

    def flatten(list_in):
        return [item for sublist in list_in for item in sublist]

    def split(list_in, length_sublists):
        list_out = [list_in[i*length_sublists:(i+1)*length_sublists] for i in xrange(len(list_in) / length_sublists)]
        return list_out

    def length(list_in):
        return len(list_in)
    
    split = pe.Node(util.Function(function=split,
                                  input_names=['list_in',
                                               'length_sublists'],
                                  output_names=['list_out']),
                    name='split')

    workflow.connect(contraster, ('varcope', flatten), sqrt, 'in_file')
    workflow.connect(sqrt, 'out_file', split, 'list_in')
    workflow.connect(contraster, ('cope', length), split, 'length_sublists')

    workflow.connect(split, 'list_out', outputspec, 'varcope')

    for field in ['z_stat', 't_stat', 'cope', 'dof']:
        workflow.connect(contraster, field, outputspec, field)

    return workflow


def create_level2_analysis_sandwich_in_limbo(name='level2_sandwich',
                                             base_dir='~/workflow_folders'):
    """Create a level 2 analysis workflow with in_limbo test
    This 
    """

    base_dir = os.path.abspath(os.path.expanduser(base_dir))

    

    inputspec = pe.Node(util.IdentityInterface(fields=['copes', 'varcopes',
                                                       'residuals', 
                                                       'design_matrices',
                                                       'dof_files',
                                                       'mask_file',
                                                       'z_threshold',
                                                       'contrast_vector',
                                                       'alpha_in_limbo']), name='inputspec')
    
    inputspec.inputs.mask_file = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')
    inputspec.inputs.alpha_in_limbo = 0.05
    
    
    """
    ORIGINAL NIPYPE CODE
    https://github.com/nipy/nipype/blob/master/nipype/workflows/fmri/fsl/estimate.py
    """
    level2 = pe.Workflow(name=name,)
    level2.base_dir = base_dir
    
    
    copemerge = pe.MapNode(interface=fsl.Merge(dimension='t'),
                           iterfield=['in_files'],
                           name="copemerge")

    varcopemerge = pe.MapNode(interface=fsl.Merge(dimension='t'),
                           iterfield=['in_files'],
                           name="varcopemerge")

    """
    Use :class:`nipype.interfaces.fsl.L2Model` to generate subject and condition
    specific level 2 model design files
    """

    level2model = pe.Node(interface=fsl.L2Model(),
                          name='l2model')

    """
    Use :class:`nipype.interfaces.fsl.FLAMEO` to estimate a second level model
    """

    flameo = pe.MapNode(interface=fsl.FLAMEO(run_mode='flame1'), name="flameo",
                        iterfield=['cope_file', 'var_cope_file'])

    def get_dofvolumes(dof_files, cope_files):
        import os
        import nibabel as nb
        import numpy as np
        img = nb.load(cope_files[0])
        out_data = np.zeros(img.get_shape())
        for i in range(out_data.shape[-1]):
            dof = np.loadtxt(dof_files[i])
            out_data[:, :, :, i] = dof
        filename = os.path.join(os.getcwd(), 'dof_file.nii.gz')
        newimg = nb.Nifti1Image(out_data, None, img.get_header())
        newimg.to_filename(filename)
        return filename

    gendof = pe.Node(util.Function(input_names=['dof_files', 'cope_files'],
                                   output_names=['dof_volume'],
                                   function=get_dofvolumes),
                     name='gendofvolume')


    level2.connect([(copemerge, gendof, [('merged_file', 'cope_files')]),
                      (copemerge, flameo, [('merged_file', 'cope_file')]),
                      (varcopemerge, flameo, [('merged_file',
                                               'var_cope_file')]),
                      (level2model, flameo, [('design_mat', 'design_file'),
                                            ('design_con', 't_con_file'),
                                            ('design_grp', 'cov_split_file')]),
                      (gendof, flameo, [('dof_volume', 'dof_var_cope_file')]),])
    
    """
    END NIPYPE CODE
    """
    
    level2.connect(inputspec, 'varcopes', varcopemerge, 'in_files')

    level2.connect(inputspec, 'mask_file', flameo, 'mask_file') 

    def get_length(in_list):
        return len(in_list)
    level2.connect(inputspec, ('copes', get_length), level2model, 'num_copes')


    level2.connect(inputspec, 'copes', copemerge, 'in_files')
    level2.connect(inputspec, 'dof_files', gendof, 'dof_files')
    level2.inputs.flameo.run_mode = 'flame1'

    smooth_est = pe.Node(fsl.SmoothEstimate(), name='smooth_est')
    smooth_est.inputs.mask_file = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

    get_one = lambda x: x[0]

    level2.connect(flameo, ('zstats', get_one), smooth_est, 'zstat_file')

    cluster = pe.Node(fsl.Cluster(), name='cluster')
    cluster.inputs.out_threshold_file = True

    level2.connect(inputspec, 'z_threshold', cluster, 'threshold')
    level2.connect(smooth_est, 'dlh', cluster, 'dlh')
    level2.connect(flameo, ('zstats', get_one), cluster, 'in_file')

    def argmin(image, threshold=0):
        import nibabel as nb
        import numpy as np
        
        image = nb.load(image)

        if np.sum(image.get_data()) == 0:
            raise Exception('No voxels above z-threshold!')

        data = np.ma.masked_less_equal(image.get_data(), threshold)
        print data.mean()
        print data.max(), data.sum()

        return np.unravel_index(np.ma.argmin(data), data.shape)

    def get_value_at(image, index):
        import nibabel as nb    
        
        return nb.load(image).get_data()[index]



    get_comparison_voxel = pe.Node(util.Function(function=argmin,
                                                 input_names=['image'],
                                                 output_names=['index']),
                                   name='get_comparison_voxel')

    level2.connect(cluster, 'threshold_file', get_comparison_voxel, 'image')

    get_covariance_comp_voxel = pe.MapNode(GetVoxelCovariance(), 
                                        iterfield=['design_matrices', 'residuals'],
                                        name='get_covariance_comp_voxel')

    level2.connect(get_comparison_voxel, 'index', get_covariance_comp_voxel, 'comp_voxel')
    level2.connect(inputspec, 'design_matrices', get_covariance_comp_voxel, 'design_matrices')
    level2.connect(inputspec, 'residuals', get_covariance_comp_voxel, 'residuals')

    get_in_limbo_weights = pe.MapNode(GetInLimboWeights(), 
                                      iterfield=['varcope',
                                                 'contrast',
                                                 'comp_voxel_covariance',],
                                      name='get_in_limbo_weights')


    level2.connect(get_comparison_voxel, 'index', get_in_limbo_weights, 'comp_voxel')

    level2.connect(inputspec, 'contrast_vector', get_in_limbo_weights, 'contrast')
    level2.connect(inputspec, 'varcopes', get_in_limbo_weights, 'varcope')
    level2.connect(get_covariance_comp_voxel, 'covariance_map', get_in_limbo_weights, 'comp_voxel_covariance')

    in_limbo_wls = pe.Node(Level2WLS(), name='in_limbo_wls')


    level2.connect(inputspec, 'copes', in_limbo_wls, 'copes')
    level2.connect(get_in_limbo_weights, 'weights', in_limbo_wls, 'weights')
    level2.connect(get_comparison_voxel, 'index', in_limbo_wls, 'comp_voxel')

    in_limbo_mapper = pe.Node(GetInLimboArea(), name='in_limbo_mapper')

    import scipy as sp
    from scipy import stats
    
    
    def get_dof(copes):
        return len(copes) - 1
    
    def get_threshold(dof, alpha):
        import scipy as sp
        from scipy import stats
        return sp.stats.t(dof).ppf(1 - alpha)
    
    get_threshold_node = pe.Node(util.Function(function=get_threshold,
                                             input_names=['dof', 'alpha'],
                                             output_names=['threshold']),
                               name='get_threshold_node')
    
    level2.connect(inputspec, ('copes', get_dof), get_threshold_node, 'dof')
    level2.connect(inputspec, 'alpha_in_limbo', get_threshold_node, 'alpha')    
    
    level2.connect(get_threshold_node, 'threshold', in_limbo_mapper, 'threshold')
    level2.connect(in_limbo_wls, 't_stats', in_limbo_mapper, 'in_limbo_t')
    level2.connect(cluster, 'threshold_file', in_limbo_mapper, 'thresholded_z')


    outputspec = pe.Node(util.IdentityInterface(fields=['z_thresholded',
                                                        'in_limbo',
                                                        'in_limbo_unactivated',]),
                         name='outputspec')

    level2.connect(in_limbo_mapper, 'in_limbo', outputspec, 'in_limbo')
    level2.connect(in_limbo_mapper, 'unactivated', outputspec, 'in_limbo_unactivated')
    level2.connect(cluster, 'threshold_file', outputspec, 'z_thresholded')

    return level2


if __name__ == '__main__':
    print 'hoi'
    level2 = create_level2_analysis_sandwich_in_limbo()
    print level2
