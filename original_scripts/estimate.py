import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
from nipype.pipeline import engine as pe
import os
import nipype.interfaces.utility as util     # utility
import sys
    
import in_limbo

workflow = pe.Workflow(name= "in_limbo_estimate")
workflow.base_dir = os.path.abspath('/home/gdholla1/Projects/in_limbo/workflow_folders')
workflow.config = {"execution": {"crashdump_dir":os.path.abspath('./crashdumps')}}

data_dir = '/home/gdholla1/data/birte/raw_nifti'

import os
import re
reg = re.compile('([a-z0-9]{4})\.des')
pps = [reg.match(fn).group(1) for fn in os.listdir('/home/gdholla1/data/birte/Designfiles_old/Duration4') if reg.match(fn) != None]


infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")

"""Here we set up iteration over all the subjects. The following line
is a particular example of the flexibility of the system.  The
``datasource`` attribute ``iterables`` tells the pipeline engine that
it should repeat the analysis on each of the items in the
``subject_list``. In the current example, the entire first level
preprocessing and estimation will be repeated for each subject
contained in subject_list.
"""

infosource.iterables = [('subject_id', pps)]

"""
Now we create a :class:`nipype.interfaces.io.DataSource` object and
fill in the information from above about the layout of our data.  The
:class:`nipype.pipeline.NodeWrapper` module wraps the interface object
and provides additional housekeeping and pipeline specific
functionality.
"""

data_dir = '/home/gdholla1/data/in_limbo_sfn/'
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                            outfields=['smoothed_file']),
                  name = 'datasource')
datasource.inputs.base_directory = data_dir
datasource.inputs.template = '*'

datasource.inputs.field_template = {'smoothed_file' : 'smoothed_file/_subject_id_%s/_fwhm_5.0/%snii2_st_mcf_smooth.nii.gz',}
datasource.inputs.template_args = dict(smoothed_file=[['subject_id', 'subject_id']])
datasource.inputs.sort_filelist = False


def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files

def get_length(l):
	return len(l)

# Speed/accuracy/neutral...
def give_onsets_and_events_birte(subject_id):
    from pandas import read_csv
    data_onsets = read_csv('/home/gdholla1/data/birte/Designfiles/Duration4/%s.des' % subject_id, sep='\t', parse_dates=False, escapechar='%')
    
    if data_onsets.onset.dtype == object:
        data_onsets['onset'] = [float(e.replace(',', '.')) for e in data_onsets['onset']]
    return list(data_onsets.event.values), list(data_onsets.onset.values)


# Get onsets, this function has to return, for every subject, a list of events 
# and a corresponding list of lists of onsets
get_onsets = pe.Node(util.Function(function=give_onsets_and_events_birte, input_names=['subject_id'], output_names=['events', 'onsets']), name='onset_getter')


# This node makes a design matrix and cuts it up into little pieces, so that
# the sandwich estimator can be used
onset_cutter = pe.Node(in_limbo.OnsetsCutter(), name='onset_cutter', overwrite=True)
onset_cutter.inputs.TR = 2.0
onset_cutter.inputs.n_blocks = 40


workflow.connect(infosource, 'subject_id', datasource, 'subject_id')
workflow.connect(infosource, 'subject_id', get_onsets, 'subject_id')
workflow.connect(datasource, ('smoothed_file', pickfirst), onset_cutter, 'data')
workflow.connect(get_onsets, 'events', onset_cutter, 'events')
workflow.connect(get_onsets, 'onsets', onset_cutter, 'onsets')


# Solves the individual cuts of the design matrix using OLS
ols_solver = pe.MapNode(in_limbo.OLSSolver(), name='ols_solver', iterfield=['data', 'design_matrix'], overwrite=False)

workflow.connect(onset_cutter, 'cutted_data', ols_solver, 'data')
workflow.connect(onset_cutter, 'design_matrices', ols_solver, 'design_matrix')


# Estimates (co)variance using sandwich estimator
sandwicher = pe.Node(in_limbo.Sandwicher(), name='sandwicher', overwrite=False)
workflow.connect(ols_solver, 'betas', sandwicher, 'betas')
workflow.connect(ols_solver, 'residuals', sandwicher, 'residuals')
workflow.connect(onset_cutter, 'design_matrices', sandwicher, 'design_matrices')


# Creates contrast out of parameter estimates
contraster = pe.Node(in_limbo.Contraster(), name='contraster', overwrite=False)

contraster.inputs.contrast = [1, -1, 0, 0, 0, 0]

workflow.connect(sandwicher, 'sandwiched_beta', contraster, 'beta')
workflow.connect(sandwicher, 'sandwiched_variance', contraster, 'variance')
workflow.connect(onset_cutter, ('design_matrices', get_length), contraster, 'n')

ds = pe.Node(nio.DataSink(), name='datasink')
ds.inputs.base_directory = '/home/gdholla1/data/in_limbo_sfn'

workflow.connect(ols_solver, 'residuals', ds, 'residuals')
workflow.connect(onset_cutter, 'design_matrices', ds, 'design_matrices')

workflow.connect(contraster, 'cope', ds, 'cope')
workflow.connect(contraster, 'varcope', ds, 'varcope')

workflow.write_graph()

nproc = 10
workflow.run(plugin='MultiProc', plugin_args={'n_procs':int(nproc)})
