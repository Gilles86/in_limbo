import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl

from nipype.workflows.fmri.fsl import create_fixed_effects_flow

import os
from .. import OnsetsCutter, OLSFitter, SandwichEstimator, SandwichContrast


def create_level1_sandwich(name='level1_sandwich', 
                           base_dir='~/workflow_folders'):

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

