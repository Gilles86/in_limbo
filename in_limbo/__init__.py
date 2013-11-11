from rpy2.robjects import r
import nibabel as nb
from rpy2.robjects import FloatVector, IntVector, ListVector
import os
from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, OutputMultiPath, isdefined
import numpy as np
import bottleneck
from nipype.interfaces.base import isdefined
import mvpa2.suite as mvpa

import nipy.modalities.fmri.design_matrix as dm
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from mvpa2.suite import save, h5load
import scipy as sp
from scipy import ndimage, spatial, stats

from rpy2.robjects import r
import nibabel as nb
from rpy2.robjects import FloatVector, IntVector, ListVector
import os
from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, OutputMultiPath, isdefined
    
import os
from tempfile import mkdtemp

class SimulateDataInputSpec(BaseInterfaceInputSpec):
    onsets = traits.List(desc='List of Onsets')
    region_centers = traits.List([[40, 40]], usedefault=True, desc='Region centers')
    region_radiuses = traits.List([8], usedefault=True, desc='Total number of trials to arrive at')
    size = traits.List([80, 80], usedefault=True, desc='Size of region')
    nscans = traits.Int(desc='Number of scans')
    SNR = traits.Float(3.75, usedefault=True, desc='Signal-to-Noise ratio')
    TR = traits.Float(2.0, usedefault=True, desc='TR')
    effect_sizes = traits.List([[1.]], usedefault=True, desc='Effect sizes per region per condition')
    durations = traits.List([[2.0]], usedefault=True, desc='Durations per region per condition')
    fading = traits.Float(0.15, usedefault=True, desc='fading from center')
    FWHM = traits.Float(1.0, usedefault=True, desc='FWHM of smoothing kernel')
    n_regions = traits.Int(1, usedefault=True, desc='Number of regions')

    #onsets, region_centers, region_radius=None, size=(75,75), nscans=100, SNR=3.75, TR=2.0, effect_sizes=None, durations=None
    #[[[5, 25], [10, 15]], [[5, 25], [10, 15]]], [[30, 30], [45, 45]], SNR=snr, effect_sizes=[[5.0, 0.0], [0.0, 0.7]])
class SimulateDataOutputSpec(TraitedSpec):
    simulated_data = traits.File(exists=True, desc="Volume containing simulated data.")


class SimulateData(BaseInterface):

    r['library']('neuRosim')

    input_spec = SimulateDataInputSpec
    output_spec = SimulateDataOutputSpec

    def _run_interface(self, runtime):

        n_conditions = len(self.inputs.onsets[0])    


        if self.inputs.n_regions == 1:
            self.onsets = [FloatVector(o) for o in self.inputs.onsets]
        else:
            self.onsets = [[FloatVector(ol_cond) for ol_cond in ol_region] for ol_region in self.inputs.onsets]

        self.region_centers = [FloatVector(rc) for rc in self.inputs.region_centers]
        #self.region_centers = self.inputs.region_centers

        if isdefined(self.inputs.nscans):        
            self.total_time = self.inputs.nscans * self.inputs.TR
        else:
            self.total_time = np.max([np.ceil((np.max(self.inputs.onsets[0]) / self.inputs.TR)) * self.inputs.TR, 140])

        design = r['simprepTemporal'](regions=self.inputs.n_regions, onsets=self.onsets, durations=self.inputs.durations, TR=self.inputs.TR, totaltime=self.total_time, effectsize=self.inputs.effect_sizes)
        spatial = r['simprepSpatial'](regions=self.inputs.n_regions, coord=self.region_centers, 
                          radius=self.inputs.region_radiuses,
                          form="sphere", 
                          fading=self.inputs.fading)

        self.sim_data = r['simVOLfmri'](design=design, image=spatial, SNR=self.inputs.SNR, 
                                        noise="mixture", 
                                        dim=IntVector(self.inputs.size), 
                                        w=FloatVector([0.05,0.1,0.01,0.09,0.05,0.7]),
                                        FWHM=self.inputs.FWHM,
                                        spat='gammaRF')

        nb.save(nb.Nifti1Image(np.array(self.sim_data), np.eye(4)), 'simulated_data.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['simulated_data'] = os.path.abspath('simulated_data.nii.gz')

        return outputs
        
        
class OnsetMakerInputSpec(BaseInterfaceInputSpec):
    n_trials = traits.Int(default_value=50, usedefault=True, desc='Total number of trials to arrive at')
    jitter = traits.Float(default_value=1., usedefault=True, desc='Amount of jitter over intervals')
    n_conditions = traits.Int(default_value=1, usedefault=True, desc='Total number of trials to arrive at')
    inter_trial_interval = traits.Float(default_value=10, usedefault=True, desc='Total length of run in seconds')
    onset_start = traits.Float(default_value=1., usedefault=True, desc='Start of first trial')

class OnsetMakerOutputSpec(TraitedSpec):
    onsets = traits.List(exists=True, desc="List of lists of onsets nConditions x nTrials")
    conditions = traits.List(exists=True, desc="List of conditions")


class OnsetMaker(BaseInterface):
    input_spec = OnsetMakerInputSpec
    output_spec = OnsetMakerOutputSpec

    def _run_interface(self, runtime):

        if self.inputs.n_conditions != 1:
            raise NotImplementedError

        self.total_length = self.inputs.n_trials * self.inputs.inter_trial_interval

        self.onsets = np.linspace(self.inputs.jitter/2, self.total_length - (self.inputs.jitter/2), self.inputs.n_trials + 1) + self.inputs.onset_start
        self.onsets = self.onsets[:-1]

        self.onsets += (np.random.random(len(self.onsets)) - 0.5) * self.inputs.jitter
        self.onsets = [list(self.onsets)]
        self.conditions = ['a'] * len(self.onsets)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['onsets'] = self.onsets
        outputs['conditions'] = self.conditions

        return outputs
        
        


class OLSGLMInputSpec(BaseInterfaceInputSpec):
    onsets = traits.List(desc='List of Onsets')
    data_files = traits.List(File(exists=True),
                              mandatory=True,
                              desc="List of functional data files")

    TR = traits.Float(2.0, usedefault=True, desc='TR')
    variance_to_use = traits.String('sandwich', usedefault=True, desc='Which variance estimator to use to calculate z-map (sandwhich or ols)')
    contrasts = traits.List([1, 0, 0], usedefault=True, desc='contrast')

class OLSGLMDataOutputSpec(TraitedSpec):
    ols_beta = traits.File(exists=True, desc="Volume containing beta-estimates")
    ols_var = traits.File(exists=True, desc="Volume containing variance-estimates (OLS).")
    sandwich_var = traits.File(exists=True, desc="Volume containing variance-estimates (sandwich).")
    z = traits.File(exists=True, desc="Volume containing variance-estimates (sandwich).")
    design_matrix = traits.File(exists=True, desc="HDF5-file containing design matrix (working model).")
    residuals = traits.File(exists=True, desc="Residuals per run")

class OLSGLM(BaseInterface):


    input_spec = OLSGLMInputSpec
    output_spec = OLSGLMDataOutputSpec

    def _run_interface(self, runtime):

        data_list = [nb.load(fn).get_data() for fn in self.inputs.data_files]
        onsets = self.inputs.onsets

        paradigm = EventRelatedParadigm(['a'] * len(onsets), onsets)
        frametimes = np.arange(0, data_list[0].shape[-1] * self.inputs.TR, self.inputs.TR)
        X, names = dm.dmtx_light(frametimes, paradigm, drift_model='polynomial',
                     hfcut=128, hrf_model='canonical')

        self.X = X

        X_T_inv = np.linalg.pinv(np.dot(X.T, X))
        calc_beta = np.dot(X_T_inv, X.T)

        # Do OLS
        mean_data = np.mean(data_list, 0)    
        self.ols_beta = np.dot(calc_beta, mean_data.reshape(np.prod(mean_data.shape[:-1]), mean_data.shape[-1]).T)
        predicted = np.dot(X, self.ols_beta).T.reshape(mean_data.shape)
        resid = mean_data - predicted
        ss = bottleneck.ss(resid, -1)
        self.ols_var = np.outer(X_T_inv, ss)

        # Create individual residuals for sandwhich:
        self.sss = np.zeros(ss.shape)
        self.residuals = []
        for data in data_list:
            beta = np.dot(calc_beta, data.reshape(np.prod(data.shape[:-1]), mean_data.shape[-1]).T)
            self.predicted = np.dot(X, self.ols_beta).T.reshape(mean_data.shape)
            resid = mean_data - self.predicted
            self.residuals.append(resid)
            self.sss += bottleneck.ss(resid, -1)

        if len(data_list) > 5:        
            self.sss = self.sss / (len(data_list) - 1)
        else:
            self.sss = self.sss / len(data_list)

        self.sandwich_var = np.outer(np.dot(calc_beta, calc_beta.T), self.sss) / len(data_list)

        self.contrasts = np.array(self.inputs.contrasts)

        self.residuals = np.array(self.residuals).swapaxes(0, -1)


        self.sandwich_var = self.sandwich_var.T.reshape(mean_data.shape[:-1] + (-1,))


        self.ols_beta = self.ols_beta.T.reshape(mean_data.shape[:-1] + (-1,))
        self.ols_var = self.ols_var.T.reshape(mean_data.shape[:-1] + (-1,))

        if self.inputs.variance_to_use == 'ols':
            self.z = (self.ols_beta[:, :, 0] / (np.sqrt(self.ols_var[:, :, 0]) / np.sqrt(len(data_list)))).squeeze()
        else:
            self.z = (self.ols_beta[:, :, 0] / (np.sqrt(self.sandwich_var[:, :, 0]) / np.sqrt(len(data_list)) )).squeeze()


        self.z = self.z.T.reshape(mean_data.shape[:-1] + (-1,))


        nb.save(nb.Nifti1Image(self.ols_beta, np.identity(4)), 'ols_beta.nii.gz') 
        nb.save(nb.Nifti1Image(self.ols_var, np.identity(4)), 'ols_var.nii.gz')
        nb.save(nb.Nifti1Image(self.sandwich_var, np.identity(4)), 'sandwich_var.nii.gz')
        nb.save(nb.Nifti1Image(self.z, np.identity(4)), 'z_%s.nii.gz' % self.inputs.variance_to_use)
        nb.save(nb.Nifti1Image(self.residuals, np.identity(4)), 'residuals.nii.gz')
        
        
        save(self.X, 'design_matrix.hdf5')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['ols_beta'] = os.path.abspath('ols_beta.nii.gz')
        outputs['ols_var'] = os.path.abspath('ols_var.nii.gz')
        outputs['sandwich_var'] = os.path.abspath('sandwich_var.nii.gz')
        outputs['z'] = os.path.abspath('z_%s.nii.gz' % self.inputs.variance_to_use)
        outputs['design_matrix'] = os.path.abspath('design_matrix.hdf5')
        outputs['residuals'] = os.path.abspath('residuals.nii.gz')

        return outputs
        
        
        import scipy as sp
        from scipy import spatial, ndimage
        import nipype.interfaces.io as nio

        # Let set up the 





class InLimboInputSpec(BaseInterfaceInputSpec):
    zmap = traits.File(desc='Zmap in NIFTI-format')
    variance = traits.File(desc='Variances in NIFTI-format')
    residuals = traits.File(desc='Residuals in NIFTI-format')    
    design_matrix = traits.File(desc='Design matrix in HDF5-format')
    clusters = traits.File(desc='Zmap in NIFTI-format')
    beta = traits.File(desc='Beta-map in NIFTI-format')
    ddof = traits.Int(desc='Degrees of freedom')
    alpha = traits.Float(0.05, usedefault=True, desc='Alpha-value for t-test')
    global_threshold = traits.Bool(True, usedefault=True, desc='Whether to use a global minimum as threshold or minimum of nearest cluster')

class InLimboOutputSpec(TraitedSpec):
    in_limbo = traits.File(exists=True, desc="In limbo-mask")
    ts = traits.File(exists=True, desc="T-values of in limbo test")


class InLimbo(BaseInterface):
    input_spec = InLimboInputSpec
    output_spec = InLimboOutputSpec

    def _run_interface(self, runtime):


        clusters = nb.load(self.inputs.clusters).get_data()
        
        if np.sum(clusters) == 0:
            nb.save(nb.Nifti1Image(np.zeros(clusters.shape), np.identity(4)), 'in_limbo.nii.gz')
            return runtime
			
        if self.inputs.global_threshold:
            clusters[clusters > 0] = 1
			
		
        # Get nearest-neighbors
        cluster_centers = [sp.ndimage.measurements.center_of_mass(clusters == c) for c in np.unique(clusters)[1:]]
        kdtree = sp.spatial.KDTree(cluster_centers)
        grid = np.mgrid[[slice(0, e) for e in clusters.shape]]
        r = kdtree.query(np.array([g.ravel() for g in grid]).T)

        assigned_clusters = r[1].reshape(clusters.shape) + 1        

        # get minimum value within cluster
        z = nb.load(self.inputs.zmap).get_data()
        X = h5load(self.inputs.design_matrix)

        # get index of minimum per cluster
        cl_min_z = {}
        for cl in np.unique(clusters[clusters > 0]):
            ind = z[clusters == cl].argmin()
            cl_min_z[cl] = tuple(np.array(np.where(clusters == cl))[:, ind])

        # get complete variances
        variances = nb.load(self.inputs.variance).get_data()
        residuals = nb.load(self.inputs.residuals).get_data()
        betas = nb.load(self.inputs.beta).get_data()

        # per cluster, get t-value
        ts = np.zeros(z.shape[:-1])
        covars = np.zeros(z.shape[:-1])

        for cl in np.unique(clusters[clusters > 0]):

            # Index of minimum in cluster
            ind_min = tuple([slice(None)] + [e for e in  cl_min_z[1]] + [0])
            var_min = variances[cl_min_z[cl]][0]
            beta_min = betas[cl_min_z[cl]][0]
            current_ind = (assigned_clusters == cl) & (clusters == 0)

            # Get covariances
            current_covars = np.zeros(np.sum(current_ind))

            for i in np.arange(residuals.shape[-1]):
                print residuals[:, current_ind, i].shape
                print residuals[ind_min].shape
                current_covars += np.dot(residuals[:, current_ind, i].squeeze().T, residuals[ind_min])

            current_covars = current_covars / residuals.shape[-1]

            pinv = np.linalg.pinv(np.dot(X.T, X))

            current_covars = np.outer(np.dot(np.dot(pinv, X.T), np.dot(X, pinv)), current_covars)[0, :]

            current_vars = variances[current_ind][:, 0]
            current_betas = betas[current_ind][:, 0]

            contrast = beta_min - current_betas 
            denom_var = np.sqrt(var_min + current_vars - 2*current_covars)
            ts[current_ind] = contrast / denom_var
            covars[current_ind] = current_covars

        threshold = sp.stats.t.ppf(1. - self.inputs.alpha, self.inputs.ddof)

        self.in_limbo = (np.abs(ts) < threshold) & (clusters == 0)

        nb.save(nb.Nifti1Image(np.array(self.in_limbo, dtype=int), np.identity(4)), 'in_limbo.nii.gz')
        nb.save(nb.Nifti1Image(np.array(ts, dtype=int), np.identity(4)), 't.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['in_limbo'] = os.path.abspath('in_limbo.nii.gz')
        outputs['ts'] = os.path.abspath('t.nii.gz')

        return outputs


class OLSSolverInputSpec(BaseInterfaceInputSpec):
    data = traits.File(usedefault=True, desc='Number of Blocks')
    design_matrix = traits.File(desc='List of onsets')

class OLSSolverOutputSpec(TraitedSpec):
    betas = traits.File(exists=True, desc="File with estimated betas")
    variances = traits.File(exists=True, desc="File with estimated variance")
    residuals = traits.File(exists=True, desc="File with residuals")


class OLSSolver(BaseInterface):
    input_spec = OLSSolverInputSpec
    output_spec = OLSSolverOutputSpec

    def _run_interface(self, runtime):

        image = nb.load(self.inputs.data)
        data = image.get_data()
        design_matrix = h5load(self.inputs.design_matrix)

        X = design_matrix

        print X.shape, data.shape


        X_T_inv = np.linalg.pinv(np.dot(X.T, X))
        calc_beta = np.dot(X_T_inv, X.T)

        beta = np.dot(calc_beta, data.reshape(np.prod(data.shape[:-1]), data.shape[-1]).T)



        predicted = np.dot(X, beta)        
        predicted = predicted.T.reshape((data.shape[:-1] + (X.shape[0],)))

        beta = beta.T.reshape(data.shape[:-1] + (X.shape[1],))
        resid = data - predicted
        ss = bottleneck.ss(resid, -1)

        #ss = resid.T.dot(resid)
		
        ols_var = np.outer(X_T_inv, ss)

        ols_var = ols_var.T.reshape(data.shape[:-1] + (ols_var.shape[0],))

        nb.save(nb.Nifti1Image(beta, image.get_affine(), image.get_header()), os.path.abspath('betas.nii.gz'))
        nb.save(nb.Nifti1Image(ols_var, image.get_affine(), image.get_header()), os.path.abspath('variances.nii.gz'))
        nb.save(nb.Nifti1Image(resid, image.get_affine(), image.get_header()), os.path.abspath('residuals.nii.gz'))


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['betas'] = os.path.abspath('betas.nii.gz')
        outputs['variances'] = os.path.abspath('variances.nii.gz')
        outputs['residuals'] = os.path.abspath('residuals.nii.gz')

        return outputs


class OnsetsCutterInputSpec(BaseInterfaceInputSpec):
    n_blocks = traits.Int(default_value=0, usedefault=True, desc='Number of Blocks')
    onsets = traits.List(desc='List of onsets')
    events = traits.List(desc='List of events')
    TR = traits.Float(default_value=2.0, desc='List of events')
    data = traits.File(desc='data_file')

class OnsetsCutterOutputSpec(TraitedSpec):
    design_matrices = traits.List(exists=True, desc="File with design matrices")
    cutted_data = traits.List(desc='List of cut-up data files')


class OnsetsCutter(BaseInterface):
    input_spec = OnsetsCutterInputSpec
    output_spec = OnsetsCutterOutputSpec

    def _run_interface(self, runtime):

        assert(len(self.inputs.onsets) == len(self.inputs.events))

        self.inputs.onsets, self.inputs.events = [list(e) for e in zip(*sorted(zip(self.inputs.onsets, self.inputs.events)))]

        if self.inputs.n_blocks == 0:
            self.inputs.n_blocks = int(np.sqrt(len(self.inputs.onsets)))


        image = nb.load(self.inputs.data)
        data = image.get_data()
        self.data_shape = image.shape

        length_block = self.data_shape[-1] / self.inputs.n_blocks


        paradigm = EventRelatedParadigm(self.inputs.events, self.inputs.onsets)
        frametimes = np.arange(0, self.data_shape[-1] * self.inputs.TR, self.inputs.TR) 
        X, names = dm.dmtx_light(frametimes, paradigm, drift_model='polynomial',
                     hfcut=128, hrf_model='canonical')

        for i in np.arange(0, self.data_shape[-1] + 1, length_block)[:-1]:
            save(X[i:i+length_block, :], os.path.abspath('./design_matrices_%s.hdf5' % (i/length_block)))            
#            nb.save(nb.Nifti1Image(data[:, :, :, i:i+length_block], image.get_affine(), image.get_header()), os.path.abspath('data_%s.nii.gz' % (i/length_block)))
            nb.save(nb.Nifti1Image(data[..., i:i+length_block], image.get_affine(), image.get_header()), os.path.abspath('data_%s.nii.gz' % (i/length_block)))


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        
        if self.data_shape[-1] % self.inputs.n_blocks == 0:
            n_blocks = self.inputs.n_blocks
        else:
            n_blocks = self.inputs.n_blocks - 1
        
        outputs['design_matrices'] = [os.path.abspath('./design_matrices_%s.hdf5' % i) for i in np.arange(n_blocks)]
        outputs['cutted_data'] = [os.path.abspath('./data_%s.nii.gz' % i) for i in np.arange(n_blocks)]

        return outputs


class SandwicherInputSpec(BaseInterfaceInputSpec):
    betas = traits.List(exists=True, desc="Files with betas")
    design_matrices = traits.List(exists=True, desc="Files with design matrices")
    residuals = traits.List(exists=True, desc="Files with design matrices")

class SandwicherOutputSpec(TraitedSpec):
    sandwiched_beta = traits.File(exists=True, desc="File with design matrices")
    sandwiched_variance = traits.File(exists=True, desc="File with design matrices")


class Sandwicher(BaseInterface):
    input_spec = SandwicherInputSpec
    output_spec = SandwicherOutputSpec

    def _run_interface(self, runtime):

        beta_images = [nb.load(beta) for beta in self.inputs.betas]
        residuals = [nb.load(r).get_data() for r in self.inputs.residuals]
        design_matrices = [h5load(dm) for dm in self.inputs.design_matrices]

        mean_beta = np.mean([beta.get_data() for beta in beta_images], 0)
        nb.save(nb.Nifti1Image(mean_beta, beta_images[0].get_affine(), beta_images[0].get_header()), 'sandwiched_beta.nii.gz')

        V = np.zeros(mean_beta.shape + (mean_beta.shape[-1], ))
        W = np.sum([bottleneck.ss(r, -1) for r in residuals], 0) / (len(residuals) - 1)

        for X, resid in zip(design_matrices, residuals):
			#W = resid.T.dot(resid)
			X_T_inv = np.linalg.pinv(np.dot(X.T, X))
			top_sandwich = np.outer(np.dot(X_T_inv, X.T), W).T.reshape((np.prod(W.shape), X_T_inv.shape[1], X.shape[0]))
			sandwich = np.dot(top_sandwich, np.dot(X, X_T_inv))            
			V = V + sandwich.reshape(V.shape)

        V = V / len(design_matrices)

        nb.save(nb.Nifti1Image(V, beta_images[0].get_affine(), beta_images[0].get_header()), 'sandwiched_variance.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['sandwiched_beta'] = os.path.abspath('./sandwiched_beta.nii.gz')
        outputs['sandwiched_variance'] = os.path.abspath('./sandwiched_variance.nii.gz')

        return outputs


class ContrasterInputSpec(BaseInterfaceInputSpec):
    contrast = traits.List(desc="List showing contrast")
    beta = traits.File(desc='Nifti file with beta estimates')
    variance = traits.File(desc='Nifti file with (co)variance estimates')
    n = traits.Int(desc='Number of replications')

class ContrasterOutputSpec(TraitedSpec):
    cope = traits.File(exists=True, desc="File with design matrices")
    varcope = traits.File(exists=True, desc="File with design matrices")
    t = traits.File(exists=True, desc="File with design matrices")
    z = traits.File(exists=True, desc="File with design matrices")


class Contraster(BaseInterface):
    input_spec = ContrasterInputSpec
    output_spec = ContrasterOutputSpec

    def _run_interface(self, runtime):

        contrast = np.array(self.inputs.contrast)

        beta_image = nb.load(self.inputs.beta)
        beta = beta_image.get_data()
        variance = nb.load(self.inputs.variance).get_data()

        cope = np.dot(contrast, beta.reshape((-1, beta.shape[-1])).T).reshape(beta.shape[:-1])

        nb.save(nb.Nifti1Image(cope, beta_image.get_affine(), beta_image.get_header()), 'cope.nii.gz')

        varcope1 = np.dot(contrast.T, variance.reshape((np.prod(cope.shape),) + variance.shape[-2:]))
        varcope = np.dot(varcope1, contrast).reshape(cope.shape)
        nb.save(nb.Nifti1Image(varcope, beta_image.get_affine(), beta_image.get_header()), 'varcope.nii.gz')       

        t = cope / np.sqrt(varcope)        
        nb.save(nb.Nifti1Image(t, beta_image.get_affine(), beta_image.get_header()), 't.nii.gz')       

        if isdefined(self.inputs.n):                
            sem = np.sqrt(varcope) / np.sqrt(self.inputs.n)
            z = cope / sem
            nb.save(nb.Nifti1Image(t, beta_image.get_affine(), beta_image.get_header()), 'z.nii.gz')            


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['cope'] = os.path.abspath('./cope.nii.gz')
        outputs['varcope'] = os.path.abspath('./varcope.nii.gz')
        outputs['t'] = os.path.abspath('./t.nii.gz')

        if isdefined(self.inputs.n):
            outputs['z'] = os.path.abspath('./z.nii.gz')

        return outputs


class ZScorerInputSpec(BaseInterfaceInputSpec):
    copes = traits.List(desc="List showing contrast")
    brain_mask = traits.File('/usr/share/fsl/4.1/data/standard/MNI152_T1_2mm_brain_mask.nii.gz', usedefault=True, desc='Nifti file with beta estimates')

class ZScorerOutputSpec(TraitedSpec):
    z = traits.File(exists=True, desc="Masked an z-scored group average")


class ZScorer(BaseInterface):
    input_spec = ZScorerInputSpec
    output_spec = ZScorerOutputSpec

    def _run_interface(self, runtime):

        cope_images = [nb.load(fn) for fn in self.inputs.copes]
        copes = [i.get_data() for i in cope_images]
        mask = nb.load(self.inputs.brain_mask).get_data()
        m = np.mean(copes, 0)
        std = np.std(copes, 0)
        sem = std / np.sqrt(len(copes))
        z = m / sem
        z[mask == 0] = 0

        nb.save(nb.Nifti1Image(z, cope_images[0].get_affine(), cope_images[0].get_header()), 'z.nii.gz')


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['z'] = os.path.abspath('./z.nii.gz')
        return outputs


class Level2SubjectNodeInputSpec(BaseInterfaceInputSpec):
    contrast = traits.List(desc='List containing contrast of interest (e.g. [1, -1])')
    design_matrices = traits.List(desc='List of design matrices of different residuals')
    residuals = traits.List(desc='List of residuals')    
    comparison_voxel = traits.Tuple(desc='Comparison_voxel')        

class Level2SubjectNodeOutputSpec(TraitedSpec):
    variances = traits.File(exists=True, desc="Contrast variances of individual voxels")
    across_voxels_variance = traits.File(exists=True, desc="Variance of contrast across voxel and comparison voxel")


# This node takes lists of residuals and design matrices of single subject,
# As well as an index of a 'comparison voxel'. It then returns the amount
# of variance in the difference of this contrast across the two
# voxels: varcope(voxel1) + varcope(voxel2)  - 2 covarcope(voxel1, voxel2)
class Level2SubjectNode(BaseInterface):
    input_spec = Level2SubjectNodeInputSpec
    output_spec = Level2SubjectNodeOutputSpec

    def _run_interface(self, runtime):

        print self.inputs.contrast

        contrast = np.array(self.inputs.contrast)
        design_matrices = [mvpa.h5load(d) for d in self.inputs.design_matrices]
        residuals = [nb.load(r).get_data() for r in self.inputs.residuals]



        filename_W_var = os.path.join(mkdtemp(), 'data.dat')        
        #W_var = np.zeros(residuals[0].shape[:-1] + residuals[0].shape[-1:] + residuals[0].shape[-1:])
        W_var = np.memmap(filename_W_var, dtype='float32', mode='w+', shape=(residuals[0].shape[:-1] + residuals[0].shape[-1:] + residuals[0].shape[-1:]))

        # Create Autocorrelation matrices, within voxels and across voxel and comparison voxel        
        
        for fd in np.arange(W_var.shape[0]):
            for i, r in enumerate(residuals):
                W_var[fd] += np.einsum('...k,...h', r[fd], r[fd])

                # Normalize without bias    
                W_var[fd] = W_var[fd] / (len(residuals) - 1)


        # Set up covariance matrices, both within voxel and across voxels
        V_var = np.zeros(residuals[0].shape[:-1] + (design_matrices[0].shape[1],) *2)


        # Fill in covariance matrix using sandwich
        
        for fd in np.arange(W_var.shape[0]):
            for i, X in enumerate(design_matrices):
                X_T_inv = np.linalg.pinv(np.dot(X.T, X))
                V_var[fd] += np.rollaxis(X_T_inv.dot(X.T).dot(W_var[fd]).dot(X).dot(X_T_inv), 0, -1)            

        # Normalize covariance matrix by number of replications
        V_var = V_var / len(design_matrices)

        variance_contrast = contrast.dot(V_var).dot(contrast.T)

        nb.save(nb.Nifti1Image(variance_contrast, nb.load(self.inputs.residuals[0]).get_affine(),  nb.load(self.inputs.residuals[0]).get_header()), 'variance_contrast.nii.gz')

        if isdefined(self.inputs.comparison_voxel):
            
            filename_W_covar = os.path.join(mkdtemp(), 'data.dat')
            #W_covar = np.zeros(residuals[0].shape[:-1] + residuals[0].shape[-1:] + residuals[0].shape[-1:])
            W_covar = np.memmap(filename_W_var, dtype='float32', mode='w+', shape=(residuals[0].shape[:-1] + residuals[0].shape[-1:] + residuals[0].shape[-1:]))
            
            for fd in np.arange(W_var.shape[0]):
                for i, r in enumerate(residuals):
                    W_covar[fd] += np.einsum('...k,...h', r[fd], r[self.inputs.comparison_voxel])
                
                W_covar[fd] = W_covar[fd] / (len(residuals) - 1)

            V_covar = np.zeros(residuals[0].shape[:-1] + (design_matrices[0].shape[1],) *2)

            for fd in np.arange(W_var.shape[0]):
                for i, X in enumerate(design_matrices):
                    X_T_inv = np.linalg.pinv(np.dot(X.T, X))
                    V_covar[fd] += np.rollaxis(X_T_inv.dot(X.T).dot(W_covar[fd]).dot(X).dot(X_T_inv), 0, -1)

            V_covar = V_covar / len(design_matrices)

            covariance_contrast = contrast.dot(V_covar).dot(contrast.T)    

            covar_filename = 'covariance_contrast_comp_voxel_%s.nii.gz' % '_'.join([str(e) for e in self.inputs.comparison_voxel])

            nb.save(nb.Nifti1Image(covariance_contrast, nb.load(self.inputs.residuals[0]).get_affine(),  nb.load(self.inputs.residuals[0]).get_header()), covar_filename)        

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['variances'] = os.path.abspath('variance_contrast.nii.gz')

        if isdefined(self.inputs.comparison_voxel):
            covar_filename = 'covariance_contrast_comp_voxel_%s.nii.gz' % '_'.join([str(e) for e in self.inputs.comparison_voxel])
            outputs['across_voxels_variance'] = os.path.abspath(covar_filename)        


        return outputs




class InLimboLevel2InputSpec(BaseInterfaceInputSpec):
    positive_zs = traits.Bool(True, usedefault=True)
    brain_mask = traits.File('/usr/share/fsl/4.1/data/standard/MNI152_T1_2mm_brain_mask.nii.gz', usedefault=True)
    thresholded_z = traits.File()
    copes = traits.List(desc='List of individual copes')
    alpha = traits.Float(0.05, usedefault=True)

class InLimboLevel2OutputSpec(TraitedSpec):
    in_limbo = traits.File()
    t = traits.File()


class InLimboLevel2(BaseInterface):
    input_spec = InLimboLevel2InputSpec
    output_spec = InLimboLevel2OutputSpec

    def _run_interface(self, runtime):

        cope_images = [nb.load(fn) for fn in self.inputs.copes]
        copes = np.array([c.get_data() for c in cope_images])
        z = nb.load(self.inputs.thresholded_z).get_data()
        brain_mask = nb.load(self.inputs.brain_mask).get_data()


        if self.inputs.positive_zs:          
            z = np.ma.masked_array(z, mask=((z<=0)))
            benchmark_voxel_index = np.where(z == np.ma.min(z))                        

        else:
            z = np.ma.masked_array(z, mask=((z>=0)))
            benchmark_voxel_index = np.where(z == np.ma.max(z[z<0]))

        t, p = stats.ttest_ind(np.array(copes[(slice(None),) + benchmark_voxel_index]), np.array(copes), 0)            
        t[~z.mask] = 0
        in_limbo = (z == 0) & brain_mask & (p > self.inputs.alpha)            
        nb.save(nb.Nifti1Image(t, cope_images[0].get_affine(), cope_images[0].get_header()), 't.nii.gz')
        nb.save(nb.Nifti1Image(in_limbo, cope_images[0].get_affine(), cope_images[0].get_header()), 'in_limbo.nii.gz')        
        nb.save(nb.Nifti1Image(z, cope_images[0].get_affine(), cope_images[0].get_header()), 'z.nii.gz')


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['t'] = os.path.abspath('./t.nii.gz')
        outputs['in_limbo'] = os.path.abspath('./in_limbo.nii.gz')
        return outputs