import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
from nipype.pipeline import engine as pe
import os
import nipype.interfaces.utility as util     # utility
import sys

cmd_folder = '/home/gdholla1/Projects/in_limbo/code/'
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
    
import in_limbo


workflow = pe.Workflow(name= "in_limbo_register")
workflow.base_dir = os.path.abspath('../../workflow_folders')
workflow.config = {"execution": {"crashdump_dir":os.path.abspath('./crashdumps')}}

data_dir = '/home/gdholla1/data/'

import os
import re
reg = re.compile('([a-z0-9]{4})\.des')
pps = [reg.match(fn).group(1) for fn in os.listdir('/home/gdholla1/data/birte/Designfiles_old/Duration4') if reg.match(fn) != None]


infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")


infosource.iterables = [('subject_id', pps)]


info = dict(smoothed_file=[['subject_id']],
            struct=[['subject_id']])



datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=['struct', 'design_matrices', 'residuals', 'cope', 'smoothed_file']),
                     name = 'datasource')
datasource.inputs.base_directory = data_dir
datasource.inputs.template = '*'

datasource.inputs.field_template = {'struct': 'birte/raw_nifti/%snii0.nii',
                                    'design_matrices' : 'in_limbo_sfn/design_matrices/_subject_id_%s/design_matrices*.hdf5',
                                    'residuals': 'in_limbo_sfn/residuals/_subject_id_%s/_ols_solver*/residuals.nii.gz',
                                    'cope': 'in_limbo_sfn/cope/_subject_id_%s/cope.nii.gz',
                                    'smoothed_file' : 'in_limbo_sfn/smoothed_file/_subject_id_%s/_fwhm_5.0/%snii2_st_mcf_smooth.nii.gz'}

datasource.inputs.template_args = dict(struct=[['subject_id']],
                                       design_matrices=[['subject_id']],
                                       residuals=[['subject_id']],
                                       cope=[['subject_id']],
                                       smoothed_file = [['subject_id', 'subject_id']])
datasource.inputs.sort_filelist = True


ds = pe.Node(nio.DataSink(), name='Data_Sink')
ds.inputs.base_directory = '/home/gdholla1/data/birte/in_limbo_sfn'

workflow.connect(infosource, 'subject_id', datasource, 'subject_id')


# BET
better = pe.Node(fsl.BET(), name='better')
workflow.connect(datasource, 'struct', better, 'in_file')

# REGISTER to get STRUCTURAL_2_MNI
get_struct_2_mni = pe.Node(fsl.FLIRT(), iterfield=['in_file'], name='get_struct_2_mni')
get_struct_2_mni.inputs.reference = '/usr/share/fsl/4.1/data/standard/MNI152_T1_2mm_brain.nii.gz'
workflow.connect(better, 'out_file', get_struct_2_mni, 'in_file')
workflow.connect(get_struct_2_mni, 'out_file', ds, 'struct_2_mni')

# INVERT TO GET MNI_2_STRUCTURAL
get_mni_2_struct = pe.Node(fsl.ConvertXFM(), iterfield=['in_file'], name='get_mni_2_struct')
get_mni_2_struct.inputs.invert_xfm = True
workflow.connect(get_struct_2_mni, 'out_matrix_file', get_mni_2_struct, 'in_file')

# REGISTER to get FUNCTIONAL_2_STRUCTURAL
get_func_2_struct = pe.Node(fsl.FLIRT(), iterfield=['in_file', 'reference'], name='get_func_2_struct')
workflow.connect(datasource, 'smoothed_file', get_func_2_struct, 'in_file')
workflow.connect(better, 'out_file', get_func_2_struct, 'reference')

# INVERT to get STRUCTURAL_2_FUNCTIONAL
get_struct_2_func = pe.Node(fsl.ConvertXFM(), iterfield=['in_file'], name='get_struct_2_func')
get_struct_2_func.inputs.invert_xfm = True
workflow.connect(get_func_2_struct, 'out_matrix_file', get_struct_2_func, 'in_file')

# CONCAT to get MNI_2_FUNCTIONAL
get_mni_2_func = pe.Node(fsl.ConvertXFM(), iterfield=['in_file', 'in_file2'], name='get_mni_2_func')
get_mni_2_func.inputs.concat_xfm = True
workflow.connect(get_mni_2_struct, 'out_file', get_mni_2_func, 'in_file')
workflow.connect(get_struct_2_func, 'out_file', get_mni_2_func, 'in_file2')

# CONCAT TO GET FUNCTIONAL2MNI
get_func_2_mni = pe.Node(fsl.ConvertXFM(), iterfield=['in_file', 'in_file2'], name='get_func_2_mni')
get_func_2_mni.inputs.concat_xfm = True
workflow.connect(get_func_2_struct, 'out_matrix_file', get_func_2_mni, 'in_file')
workflow.connect(get_struct_2_mni, 'out_matrix_file', get_func_2_mni, 'in_file2')
workflow.connect(get_func_2_mni, 'out_file', ds, 'mats.functional_2_mni')


# Register beta-estimates
register_cope = pe.Node(fsl.ApplyXfm(), name='register_cope')
register_cope.inputs.reference = '/usr/share/fsl/4.1/data/standard/MNI152_T1_2mm_brain.nii.gz'
register_cope.inputs.apply_xfm = True
workflow.connect(get_func_2_mni, 'out_file', register_cope, 'in_matrix_file')
workflow.connect(datasource, 'cope', register_cope, 'in_file')
workflow.connect(register_cope, 'out_file', ds, 'registered_copes')


# Register residuals
register_residuals = pe.MapNode(fsl.ApplyXfm(), iterfield=['in_file'], name='register_residuals')
register_residuals.inputs.reference = '/usr/share/fsl/4.1/data/standard/MNI152_T1_2mm_brain.nii.gz'
register_residuals.inputs.apply_xfm = True
workflow.connect(get_func_2_mni, 'out_file', register_residuals, 'in_matrix_file')
workflow.connect(datasource, 'residuals', register_residuals, 'in_file')
workflow.connect(register_residuals, 'out_file', ds, 'registered_residuals')

nproc = 10
workflow.run(plugin='MultiProc', plugin_args={'n_procs':int(nproc)})