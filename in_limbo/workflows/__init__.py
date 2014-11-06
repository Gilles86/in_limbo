import ..sandwich
import nipype.algorithms.modelgen as model
import pandas
import nipype.pipeline.engine as pe
from nipype.interfaces.base import Bunch

import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl

from nipype.workflows.fmri.fsl import create_fixed_effects_flow

def create_level1_sandwich(name='level1_sandwich', 
                           base_dir='~/workflow_folders'):


    workflow = pe.Workflow(name=name, base_dir=os.path.abspath(base_dir))


    inputspec = pe.Node(util.IdentityInterface(fields=['functional_runs',
                                                       'session_info',
                                                       'contrasts', 'TR']),
                        name='identity')

    onsets_cutter = pe.MapNode(sandwich.OnsetsCutter(),
                               iterfield=['session_info'],
                               name='onsets_cutter')
    workflow.connect(inputspec, 'session_info', onsets_cutter, 'session_info')

    ols_esimator = pe.MapNode(sandwich.OLSFitter(), 
                              iterfield=['X', 'data'],
                              name='ols_esimator')

    workflow.connect(onsets_cutter, 'X', ols_esimator, 'X')
    workflow.connect(selector, ('func', first_element), ols_esimator, 'data')

    sandwich_estimator = pe.MapNode(sandwich.SandwichEstimator(), 
                                    iterfield=['beta', 'design_matrices', 'cutted_data'],
                                    name='sandwich_estimator')
    workflow.connect(ols_esimator, 'beta', sandwich_estimator, 'beta')
    workflow.connect(onsets_cutter, 'design_matrices', sandwich_estimator, 'design_matrices')
    workflow.connect(onsets_cutter, 'cutted_data', sandwich_estimator, 'cutted_data')


    contraster = pe.Node(sandwich.SandwichContrast(), 
                         iterfield=['beta', 'variance', 'q', 'session_info'],
                         name='contraster')
    workflow.connect(inputspec, 'contrasts', contraster, 'contrasts')
    workflow.connect(ols_esimator, 'beta', contraster, 'beta')
    workflow.connect(sandwich_estimator, 'variance', contraster, 'variance')
    workflow.connect(onsets_cutter, 'q', contraster, 'q')
    workflow.connect(spec, 'session_info', contraster, 'session_info')


    outputspec = pe.Node(util.IdentityInterface(fields=['beta_ols',
                                                        'beta_variance',
                                                        'residuals',
                                                        'design_matrices',
                                                        'dof', 'z_stat',
                                                        't_stat', 'cope',
                                                        'varcope', 'dof',]
                                                name='outputspec')

    workflow.connect(ols_esimator, 'beta', outputspec, 'beta_ols')
    workflow.connect(ols_esimator, 'variance', outputspec, 'beta_variance')
    workflow.connect(sandwich_estimator, 'residuals', outputspec, 'residuals')
    workflow.connect(onsets_cutter, 'design_matrices', outputspec, 'design_matrices')

    for field in ['z_stat', 't_stat', 'cope', 'varcope', 'dof']:
        workflow.connect(contraster, field, outputspec, field)

    workflow.run(plugin='MultiProc', plugin_args={'n_procs':int(6)})
