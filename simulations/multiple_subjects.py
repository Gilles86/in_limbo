from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util
import os

from interfaces import OnsetMaker, SimulateData, OLSGLM, InLimbo

from nipype.interfaces import fsl
import nipype.interfaces.utility as util
import numpy as np
import nipype.interfaces.io as nio



# Set up workflow
workflow = pe.Workflow(name="simulate_multiple_subjects",
base_dir=os.path.join(os.environ['HOME'], 'workflow_folders'))
workflow.base_dir = base_dir


def simulate_subject(spatial_jitter, radius_jitter, effect_size_jitter, snr_jitter=1, standard_snr=2.0):
    import numpy as np
    rand_nrs = np.random.randn(4)
    
    region_centers = [[25 + rand_nrs[0] * spatial_jitter, 25 + rand_nrs[1] * spatial_jitter],
                      [55 + rand_nrs[2] * spatial_jitter, 55 + rand_nrs[3] * spatial_jitter]]
    
    
    rand_nrs = np.random.randn(2)
    radiuses = [15 + rand_nrs[0] * radius_jitter, 10 + rand_nrs[1] * radius_jitter]
    
    rand_nrs = np.random.randn(2)
    effect_sizes = [[30.0 + rand_nrs[0] * effect_size_jitter], [25.0  + rand_nrs[1] * effect_size_jitter]]
    
    rand_nrs = np.random.randn(1)
    snr = standard_snr + np.max(rand_nrs[0] * snr_jitter, 0.1)
    
    return region_centers, radiuses, effect_sizes, snr


subject_simulator = pe.MapNode(interface=util.Function(function=simulate_subject, input_names=['spatial_jitter', 'radius_jitter', 'effect_size_jitter'], output_names=['region_centers', 'radiuses', 'effect_sizes', 'snr']), name='subject_simulator', iterfield=['spatial_jitter'], overwrite=True)
subject_simulator.inputs.spatial_jitter = [3] * 25
subject_simulator.inputs.radius_jitter = 5
subject_simulator.inputs.effect_size_jitter = 5


# Set up onset creator
onset_maker = pe.Node(interface=OnsetMaker(), name='onset_maker', overwrite=False)
onset_maker.inputs.jitter = 1
onset_maker.inputs.n_trials = 50

def double_list(input):
    return [input, input]


def function_repeat(list_in, compare_list):
    if type(list_in) is not list:
        list_in = [list_in]

    return list_in * (len(compare_list)/len(list_in))

repeater = pe.Node(interface=util.Function(function=function_repeat, input_names=['list_in', 'compare_list'], output_names=['repeated_list']), name="repeater_node")

# Set up simulation
simulator = pe.MapNode(interface=SimulateData(), name='simulator', iterfield=['region_centers', 'region_radiuses', 'effect_sizes', 'SNR'])

simulator.inputs.durations = [[2.0], [2.0]]
#simulator.inputs.nscans = 350*5 + 10
simulator.iterables = [('fading', [0.01])]
#simulator.inputs.SNR = 2.1
simulator.inputs.FWHM = 3
simulator.inputs.n_regions = 2



# simulator.iterables = [('SNR', [1.0, 2.0, 3.0]), ('FWHM', np.arange(1, 11))]



glm = pe.MapNode(interface=OLSGLM(), name='GLM', overwrite=False, iterfield=['onsets', 'data_files'])


smooth_est = pe.MapNode(interface=fsl.SmoothEstimate(), name='smoothness_estimator', iterfield=['zstat_file', 'mask_file'])

def pickfirst(list):
    return list[0]

def listify(element):
    
    if type(element) == list:
        return [[e] for e in element]
    else:
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


masker = pe.MapNode(interface=util.Function(input_names=['data_file'], output_names=['mask'], function=mask_all), name='masker', iterfield=['data_file'])


datasink = pe.Node(interface=nio.DataSink(), name='Datasink')
datasink.inputs.base_directory = '/home/gdholla1/Projects/in_limbo/datasink_multiple_subjects'

cluster = pe.MapNode(interface=fsl.Cluster(), name='cluster', iterfield=['dlh', 'volume', 'in_file'])
cluster.inputs.threshold = 2.3
cluster.inputs.pthreshold = 0.05
cluster.inputs.out_index_file = True
cluster.inputs.out_localmax_txt_file = True

in_limbo = pe.MapNode(interface=InLimbo(), name='in_limbo', iterfield=['zmap', 'variance', 'residuals', 'design_matrix', 'clusters', 'beta'])
in_limbo.inputs.ddof = 49

workflow.connect(glm, 'z', in_limbo, 'zmap')
workflow.connect(glm, 'sandwich_var', in_limbo, 'variance')
workflow.connect(glm, 'residuals', in_limbo, 'residuals')
workflow.connect(glm, 'design_matrix', in_limbo, 'design_matrix')
workflow.connect(cluster, 'index_file', in_limbo, 'clusters')
workflow.connect(glm, 'ols_beta', in_limbo, 'beta')

workflow.connect(onset_maker, ('onsets', double_list), simulator, 'onsets')

workflow.connect(onset_maker, 'onsets', repeater, 'list_in')
workflow.connect(simulator, 'simulated_data' , repeater, 'compare_list')

workflow.connect(repeater, 'repeated_list', glm, 'onsets')
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

workflow.connect([(subject_simulator, simulator,
                     [('region_centers', 'region_centers'),
                      ('radiuses', 'region_radiuses'),
                      ('effect_sizes', 'effect_sizes'),
                      ('snr', 'SNR')]
                   ),
                   (glm, datasink, [('ols_beta', 'ols_beta'), ('sandwich_var', 'sandwich_var')])
                 ])

workflow.write_graph()
workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 10})


