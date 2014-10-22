from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util
import os
import sys
lib_path = os.path.abspath('../')
sys.path.append(lib_path)

import in_limbo

from in_limbo import OnsetMaker, SimulateData, OLSGLM, InLimbo

from nipype.interfaces import fsl
import nipype.interfaces.utility as util
import numpy as np
import nipype.interfaces.io as nio



# Set up workflow
workflow = pe.Workflow(name= "simulation_multiple_regions_v2")
base_dir=os.path.join(os.environ['HOME'], 'workflow_folders'))
workflow.base_dir = base_dir

identity = pe.Node(util.IdentityInterface(fields=['n'], mandatory_inputs=False), name='N-identity')
identity.iterables = [('n', np.array(np.linspace(5, 350, 4), dtype=int))]
# identity.iterables = [('n', 50)]
# identity.iterables = [('n', [50, 75, 100, 150, 200])]

# Set up onset creator
onset_maker = pe.Node(interface=OnsetMaker(), name='onset_maker', overwrite=False)
onset_maker.inputs.jitter = 1

def double_list(input):
    return [input, input]


# Set up simulation
simulator = pe.Node(interface=SimulateData(), name='simulator')
simulator.inputs.region_centers = [[20, 20], [60, 60]]
simulator.inputs.region_radiuses = [15, 10]
simulator.inputs.effect_sizes = [[30.0], [25.0]]
simulator.inputs.durations = [[2.0], [2.0]]
#simulator.inputs.nscans = 350*5 + 10
# simulator.inputs.SNR = 2.1
# simulator.inputs.FWHM = 3
simulator.inputs.n_regions = 2

simulator.iterables = [('SNR', [1.0, 3.0, 10.0, 50.0]), ('FWHM', [1.0]), ('fading', [0.005, 0.01, 0.1])]



glm = pe.Node(interface=OLSGLM(), name='GLM', overwrite=False)


smooth_est = pe.Node(interface=fsl.SmoothEstimate(), name='smoothness_estimator')

def pickfirst(list):
    return list[0]

def listify(element):
    return [element]

def minus_one(e):
    return e- 1


def listify_list_elements(l):
    return [[e] for e in l]

def mask_all(data_file, n=2):    
    import os
    import nibabel as nb
    import numpy as np
    image = nb.load(data_file) 
    data = image.get_data()
    mask = np.ones(data.shape[:n])
    nb.save(nb.Nifti1Image(mask, image.get_affine()), 'mask.nii.gz')
    return os.path.abspath('mask.nii.gz')


masker = pe.Node(interface=util.Function(input_names=['data_file'], output_names=['mask'], function=mask_all), name='masker')


datasink = pe.Node(interface=nio.DataSink(), name='Datasink')
datasink.inputs.base_directory = '/home/gdholla1/Projects/in_limbo/datasink_multiple_regions_2'

cluster = pe.Node(interface=fsl.Cluster(), name='cluster')
cluster.inputs.threshold = 2.3
cluster.inputs.pthreshold = 0.05
cluster.inputs.out_index_file = True
cluster.inputs.out_localmax_txt_file = True

in_limbo = pe.Node(interface=InLimbo(), name='in_limbo')

workflow.connect(glm, 'z', in_limbo, 'zmap')
workflow.connect(glm, 'sandwich_var', in_limbo, 'variance')
workflow.connect(glm, 'residuals', in_limbo, 'residuals')
workflow.connect(glm, 'design_matrix', in_limbo, 'design_matrix')
workflow.connect(cluster, 'index_file', in_limbo, 'clusters')
workflow.connect(glm, 'ols_beta', in_limbo, 'beta')
workflow.connect(identity, ('n', minus_one), in_limbo, 'ddof')


workflow.connect(identity, 'n', onset_maker, 'n_trials')

workflow.connect(onset_maker, ('onsets', double_list), simulator, 'onsets')
workflow.connect(onset_maker, ('onsets', pickfirst), glm, 'onsets')
workflow.connect(simulator, ('simulated_data', listify), glm, 'data_files')
workflow.connect(simulator, 'simulated_data', masker, 'data_file')
workflow.connect(glm, 'z', smooth_est, 'zstat_file')
workflow.connect(masker, 'mask', smooth_est, 'mask_file')

workflow.connect(glm, 'z', cluster, 'in_file')
workflow.connect(smooth_est, 'dlh', cluster, 'dlh')
workflow.connect(smooth_est, 'volume', cluster, 'volume')

workflow.connect(in_limbo, 'in_limbo', datasink, 'in_limbo')
workflow.connect(glm, 'z', datasink, 'z')
workflow.connect(cluster, 'index_file', datasink, 'clusters')
# workflow.connect(glm, 'design_matrix', datasink, 'design_matrix')
# workflow.connect(glm, 'residuals', datasink, 'residuals')
# workflow.connect(glm, 'ols_beta', datasink, 'betas')

# workflow.write_graph()
workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 10})


