import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
from nipype.pipeline import engine as pe
import os
import nipype.interfaces.utility as util     # utility

workflow = pe.Workflow(name= "in_limbo_preprocessing")
workflow.base_dir = os.path.abspath('../workflow_folders')
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
info = dict(func=[['subject_id']],
            struct=[['subject_id']])

datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=['func', 'struct']),
                     name = 'datasource')
datasource.inputs.base_directory = data_dir
datasource.inputs.template = '*'

datasource.inputs.field_template = {'func' : '%snii2.nii',
                            'struct': '%snii0.nii'}
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = False


motion_correct = pe.Node(interface=fsl.MCFLIRT(save_mats = True,
                                                  save_plots = True),
                            name='realign')



slicetimer = pe.Node(interface=fsl.SliceTimer(),
                       name='slicetimer')

slicetimer.inputs.time_repetition = 2.


smoother = pe.Node(interface=fsl.utils.Smooth(), name='smoother')
smoother.iterables = [('fwhm', [5.0])]


ds = pe.Node(nio.DataSink(), name='datasink')
ds.inputs.base_directory = '/home/gdholla1/data/in_limbo_sfn'

workflow.connect(infosource, 'subject_id', datasource, 'subject_id')
workflow.connect(datasource, 'func', slicetimer, 'in_file')
workflow.connect(slicetimer, 'slice_time_corrected_file', motion_correct, 'in_file')
workflow.connect(motion_correct, 'out_file', smoother, 'in_file')

workflow.connect(smoother, 'smoothed_file', ds, 'smoothed_file')


nproc = 10
workflow.run(plugin='MultiProc', plugin_args={'n_procs':int(nproc)})