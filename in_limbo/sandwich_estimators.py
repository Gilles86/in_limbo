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

class OnsetsCutterInputSpec(BaseInterfaceInputSpec):
    q = traits.Int(desc='Number of blocks')
    session_info = traits.Any(desc='session info')
    TR = traits.Float(default_value=2.0, desc='List of events')

class OnsetsCutterOutputSpec(TraitedSpec):
    design_matrices = traits.List(exists=True, desc="File with design matrices")
    cutted_data = traits.List(desc='List of cut-up data files')
    X = traits.File(desc="Complete design matrix")
    q = traits.Int(desc='Number of blocks')


class OnsetsCutter(BaseInterface):
    input_spec = OnsetsCutterInputSpec
    output_spec = OnsetsCutterOutputSpec

    def _run_interface(self, runtime):

        
        session_info = self.inputs.session_info
        TR = self.inputs.TR
        
       
        data = nb.load(session_info['scans'])        
        
        if not isdefined(self.inputs.q):
            self.inputs.q = int(np.sqrt(data.shape[-1]))

        
        q = self.inputs.q
        
        events = [[d['name']] * len(d['onset']) for d in session_info['cond']]
        onsets = [d['onset'] for d in session_info['cond']]

        import itertools
        events = list(itertools.chain(*events))
        onsets = list(itertools.chain(*onsets))

        paradigm = EventRelatedParadigm(events, onsets)

        frametimes = np.arange(0, data.shape[-1] * TR, TR) 
        X, names = dm.dmtx_light(frametimes, paradigm, drift_model='polynomial',
                  hfcut=128, hrf_model='canonical')
        
        np.save(os.path.abspath('X.npy'), X)

        cut_size = data.shape[-1] / (q)
        cuts = np.arange(0, data.shape[-1]+1, cut_size)

        data_cut = np.array([data.get_data()[..., cuts[i]:cuts[i+1]] for i in np.arange(q)])
        X_cut = np.array([X[cuts[i]:cuts[i+1], :] for i in np.arange(q)])
        
        for i, (d, fn) in enumerate(zip(data_cut, self._gen_fnames('cutted_data'))):
            nb.save(nb.Nifti1Image(d, data.get_affine()), fn)

        for i, (x, fn) in enumerate(zip(X_cut, self._gen_fnames('design_matrices'))):
            np.save(fn, x)


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        
        _, fn, ext = filemanip.split_filename(self.inputs.session_info['scans'])
        
        
        for field in ['design_matrices', 'cutted_data']:
            outputs[field] = self._gen_fnames(field)

        outputs['q'] = self.inputs.q
        outputs['X'] = os.path.abspath('X.npy')

        return outputs
    
    def _gen_fnames(self, field):       
        _, fn, ext = filemanip.split_filename(self.inputs.session_info['scans'])
        
        if field == 'cutted_data':
            return [os.path.abspath('%s_cut%d%s' % (fn, i+1, ext)) for i in np.arange(self.inputs.q)]
        elif field == 'design_matrices':
            return [os.path.abspath('design_matrix_cut%d.npy') % (i+1) for i in np.arange(self.inputs.q)]
        


class SandwichEstimatorInputSpec(BaseInterfaceInputSpec):
    design_matrices = traits.List(desc='Files containing design matrices')
    beta = traits.File(exists=True)
    cutted_data = traits.List(desc='Cutted data')    


class SandwichEstimatorOutputSpec(TraitedSpec):
    variance = traits.File(exists=True, desc='File with variance')
    residuals = traits.List(traits.File, desc='Residuals per desin matrix')


class SandwichEstimator(BaseInterface):
    input_spec = SandwichEstimatorInputSpec
    output_spec = SandwichEstimatorOutputSpec

    def _run_interface(self, runtime):
        
        q = len(self.inputs.design_matrices)
        print q
        
        data_cut = np.array([nb.load(d).get_data().astype(float) # Make sure that you don't load int16, that goes completely wrong with large values! 
                             for d in self.inputs.cutted_data]) 
        X_cut = np.array([np.load(d) for d in self.inputs.design_matrices])

        #beta = np.zeros((data_cut.shape[1:-1] + X_cut.shape[-1:] ))  # (size_x, size_y, size_z, num_regressors)
        predicted = np.zeros_like(data_cut)
        residuals = np.zeros_like(data_cut)

        #for i, x in enumerate(X_cut):
            #beta += np.einsum('ij,...j->...i', np.linalg.pinv(x.T.dot(x)).dot(x.T), data_cut[i])

        #beta /= q

        beta = nb.load(self.inputs.beta).get_data()


        for i, x in enumerate(X_cut):
            # Now we can get predicted timecourses
            predicted[i] = np.tensordot(beta, x, (-1, 1))

            # ... and residuals
            residuals[i] = data_cut[i] - predicted[i]


        W = np.zeros((residuals.shape[-1],) * 2 + data_cut.shape[1:-1]) # (length_block, length_block, width, height)

        for residual in residuals:
            W += np.einsum('...i, ...j->ij...', residual, residual) # do outer product over last two dimensions, broadcast the rest

        W /= (q - 1)
        
        V = np.zeros((X_cut.shape[-1],) * 2 + data_cut.shape[1:-1]) # (num_regresor, num_regressor, width, height, ...)
        

        for i, x in enumerate(X_cut):
            
            print i

            sandwich1 = np.linalg.pinv(x.T.dot(x)).dot(x.T)    
            sandwich2 = x.dot(np.linalg.pinv(x.T.dot(x)))

            # Apply first part sandwich
            v = np.tensordot(sandwich1, W, axes=((1), (0)))

            # Apply second part
            v = np.tensordot(v, sandwich2, (1, 0))
            

            # Roll axis to get correct shape
            v = np.rollaxis(v, -1, 1)
            

            V += v

        V /= q
        
        print V.shape
        
        #nb.save(nb.Nifti1Image(beta, 
                               #nb.load(self.inputs.cutted_data[0]).get_affine(),),
                #os.path.abspath('beta.nii.gz'))
        
        nb.save(nb.Nifti1Image(V, 
                               np.identity(4),),
                os.path.abspath('variance.nii.gz'))        

        for i, residual in enumerate(residuals):
            nb.save(nb.Nifti1Image(residual, 
                                   nb.load(self.inputs.cutted_data[0]).get_affine(),),
                    os.path.abspath('residuals_%d.nii.gz' % (i+1)))        

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        
        #outputs['beta'] = os.path.abspath('beta.nii.gz')
        outputs['variance'] = os.path.abspath('variance.nii.gz')
        outputs['residuals'] = [os.path.abspath('residuals_%d.nii.gz' % (i+1)) for i in np.arange(len(self.inputs.design_matrices))]

        return outputs


class SandwichContrastInputSpec(BaseInterfaceInputSpec):
    contrasts = traits.List(desc='List of contrasts.')
    q = traits.Int(desc='Number of cuts')
    session_info = traits.Any(desc='session info')
    beta = traits.File('File containing parameter estimates')
    variance = traits.File('File containing variance estimates')    


class SandwichContrastOutputSpec(TraitedSpec):
    t_stat = traits.List(traits.File, exists=True, desc="t-stats of contrasts")
    z_stat = traits.List(traits.File, exists=True, desc='Z-stats of contrasts')
    cope = traits.List(traits.File, exists=True, desc='File with parameter estimates of contrast sizes')
    varcope = traits.List(traits.File, exists=True, desc='File with variance of contrasts')
    dof = traits.File(exists=True)
    contrast_vector = traits.List(traits.File(exists=True))


class SandwichContrast(BaseInterface):
    input_spec = SandwichContrastInputSpec
    output_spec = SandwichContrastOutputSpec

    def _run_interface(self, runtime):
        
        
        contrasts = self.inputs.contrasts
        
        q = self.inputs.q
        dof = q - len(contrasts[0])
        
        beta_image = nb.load(self.inputs.beta)
        
        session_info = self.inputs.session_info
        beta = beta_image.get_data()
        variance = nb.load(self.inputs.variance).get_data()

        ts = []
        zs = []
        copes = []
        varcopes = []

        cvs = []

        for contrast in contrasts:
            cv = np.zeros((1, beta.shape[-1]))
            for key, value in zip(*contrast[2:]):
                for i, cond in enumerate(session_info['cond']):
                    if cond['name'] == key:
                        cv[0, i] = value

            cvs.append(cv)
            
            cope = np.tensordot(cv, beta, (1, -1)).squeeze()
            varcope = np.tensordot(np.tensordot(cv, variance, (1, 0)), cv, (1, 1)).squeeze()
            
            copes.append(cope)
            varcopes.append(varcope)
            
            t = cope / np.sqrt(varcope)
            ts.append(t)
            
            z = sp.stats.norm.ppf(sp.stats.t(dof).cdf(t))
            zs.append(z)
            
        for cope, fn in zip(copes, self._gen_fnames('cope')):
            nb.save(nb.Nifti1Image(cope, beta_image.get_affine()), fn)                        
            
        for varcope, fn in zip(varcopes, self._gen_fnames('varcope')):
            nb.save(nb.Nifti1Image(varcope, beta_image.get_affine()), fn)

        for t, fn in zip(ts, self._gen_fnames('t_stat')):
            nb.save(nb.Nifti1Image(t, beta_image.get_affine()), fn)
            
        for z, fn in zip(ts, self._gen_fnames('z_stat')):
            nb.save(nb.Nifti1Image(t, beta_image.get_affine()), fn)            
           
        for cv, fn in zip(cvs, self._gen_fnames('contrast_vector')):
            np.savetxt(fn, cv)

        np.savetxt(os.path.abspath('dof'), [dof], fmt='%d')


        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        
        for field in ['cope', 'varcope', 't_stat', 'z_stat', 'contrast_vector']:
            outputs[field] = self._gen_fnames(field)

        outputs['dof'] = os.path.abspath('dof')

        return outputs
    
    
    def _gen_fnames(self, field):       
       
        fns = []

        if field == 'contrast_vector':
            ext = 'txt'
        else:
            ext = 'nii.gz'
        
        for i in np.arange(len(self.inputs.contrasts)):
            fns.append(os.path.abspath('%s%d.%s' % (field, i+1, ext)))
            
        return fns    




class OLSFitterInputSpec(BaseInterfaceInputSpec):
    X = traits.File(desc='Entire design matrix')
    data = traits.File(desc='data')    


class OLSFitterOutputSpec(TraitedSpec):
    beta = traits.File(exists=True, desc="File with design matrices")
    variance = traits.File(exists=True, desc="File with design matrices")
    residuals = traits.File(exists=True, desc="File with design matrices")

    

class OLSFitter(BaseInterface):
    input_spec = OLSFitterInputSpec
    output_spec = OLSFitterOutputSpec

    def _run_interface(self, runtime):
        
        X = np.load(self.inputs.X)  
        data_image = nb.load(self.inputs.data) 
        data = data_image.get_data() 
        beta = np.einsum('ij,...j->...i', np.linalg.pinv(X.T.dot(X)).dot(X.T), data)
        
#         beta = np.rollaxis(beta, -1, 0)
        
#         print beta.shape
        
        predicted = np.tensordot(beta, X, (-1, 1))

        residuals = data - predicted

        s2 = np.einsum('...i,...i->...', residuals, residuals)
        s2 /= residuals.shape[0] - X.shape[1]

        V = s2[..., np.newaxis, np.newaxis] * np.linalg.pinv(X.T.dot(X))
        
        
        nb.save(nb.Nifti1Image(beta, data_image.get_affine()), os.path.abspath('beta.nii.gz'))
        nb.save(nb.Nifti1Image(V, data_image.get_affine()), os.path.abspath('V.nii.gz'))
        nb.save(nb.Nifti1Image(residuals, data_image.get_affine()), os.path.abspath('residuals.nii.gz'))   

        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        
        outputs['beta'] = os.path.abspath('beta.nii.gz')
        outputs['variance'] = os.path.abspath('V.nii.gz')
        outputs['residuals'] = os.path.abspath('residuals.nii.gz')

        return outputs


class GetVoxelCovarianceSandwichInputSpec(BaseInterfaceInputSpec):
    design_matrices = traits.List(desc='List of design matrices')
    comp_voxel = traits.Tuple(desc='Comparison voxel to make covariance matrix against')
    residuals = traits.List(traits.File, desc='List of sandwich residuals')
    

class GetVoxelCovarianceSandwichOutputSpec(TraitedSpec):
    covariance_map = traits.File(exists=True, desc="Covariance map (regressor x regressor x (image size))")


class GetVoxelCovariance(BaseInterface):
    input_spec = GetVoxelCovarianceSandwichInputSpec
    output_spec = GetVoxelCovarianceSandwichOutputSpec

    def _run_interface(self, runtime):
        
        # Residuals is a lit of nifti-images, data is not loaded yet
        residuals = [nb.load(r) for r in self.inputs.residuals]
        
        # Design matrices are the design matrices of the different 
        # replications.
        design_matrices = [np.load(dm) for dm in self.inputs.design_matrices]
        
        # Comparison voxel
        comp_index = self.inputs.comp_voxel
        
        # Get number of replications
        q = len(residuals)

        # Allocate memory for W_ij matrix
        W_ij = np.zeros((residuals[0].shape[-1],) * 2 + residuals[0].shape[:-1]) # (length_block, length_block, (image dimensions))


        # # Create covariance matrices for every residual
        for i, residual in enumerate(residuals):
            print 'residual %d' % i, residual.get_data().sum(), 
            W_ij += np.einsum('...i, ...j->ij...', residual.get_data()[comp_index], residual.get_data()) # do outer product over last two dimensions, broadcast the rest
            print W_ij.sum()

        # # Normalize
        W_ij /= (q - 1)

        # # Allocate memory for the covariance matrix of every voxel with the comparison voxel
        V_ij = np.zeros((design_matrices[0].shape[-1],) * 2 + residuals[0].shape[:-1]) # (num_regresor, num_regressor, width, height)

        for i, x in enumerate(design_matrices):
            print i
            print V_ij.sum()

            sandwich1 = np.linalg.pinv(x.T.dot(x)).dot(x.T)    
            sandwich2 = x.dot(np.linalg.pinv(x.T.dot(x)))

            # Apply first part sandwich
            v = np.tensordot(sandwich1, W_ij, axes=((1), (0)))

            # Apply second part
            v = np.tensordot(v, sandwich2, (1, 0))

            # Roll axis to get correct shape
            v = np.rollaxis(v, -1, 1)

            V_ij += v

        V_ij /= q
        
        nb.save(nb.Nifti1Image(V_ij, residuals[0].get_affine(),),
                               os.path.abspath('covariance_map_%s.nii.gz' % '_'.join(str(e) for e in self.inputs.comp_voxel)))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        
        outputs['covariance_map'] = os.path.abspath('covariance_map_%s.nii.gz' % '_'.join(str(e) for e in self.inputs.comp_voxel))

        return outputs



class Level2WLSInputSpec(BaseInterfaceInputSpec):
    weights = traits.List(traits.File, desc='List of weight maps')
    copes = traits.List(desc='List of contrast estimates')
    comp_voxel = traits.Tuple(desc='Comparison voxel to make covariance matrix against')
    

class Level2WLSOutputSpec(TraitedSpec):
    t_stats = traits.File(exists=True, desc="Covariance map (regressor x regressor x (image size))")
    z_stats = traits.File(exists=True, desc="Covariance map (regressor x regressor x (image size))")    


class Level2WLS(BaseInterface):
    input_spec = Level2WLSInputSpec
    output_spec = Level2WLSOutputSpec

    def _run_interface(self, runtime):
        
        
        copes = np.array([nb.load(cope).get_data() for cope in self.inputs.copes])
        print copes.shape
        
        Z = (copes[(Ellipsis,) + self.inputs.comp_voxel] - np.array(copes)[:, ...].T).T # Transform to facilitate broadcasting            
        W = np.array([nb.load(w).get_data() for w in self.inputs.weights])
        print W.shape
        
        G = np.ones((W.shape[0], 1))
        
        beta = np.einsum('i...,i...->...i', G, W)
        beta = 1. / np.einsum('...i,i...->...', beta, G)
        beta = np.einsum('...,i...->...i', beta, G)
        beta = np.einsum('...i,i...->i...', beta, W)
        beta = np.einsum('i...,i...->...', beta, Z)

        residuals = (Z - beta)
        s2 = np.einsum('i...,i...->...i', residuals, W)
        s2 = np.einsum('...i,i...->...', s2, residuals)
        s2 /= (residuals.shape[0] - 1)

        wls_V = 1. / W.sum(0) # inv(G.T.dot(W).dot(G))
        wls_V *= s2
        
        t = beta / np.sqrt(wls_V)
        
        dof = len(copes) - 1
        z = sp.stats.norm.ppf(sp.stats.t(dof).cdf(t))
        
        nb.save(nb.Nifti1Image(t, nb.load(self.inputs.copes[0]).get_affine()),
                               os.path.abspath('t_stats.nii.gz'))
        
        nb.save(nb.Nifti1Image(z, nb.load(self.inputs.copes[0]).get_affine()),
                               os.path.abspath('z_stats.nii.gz'))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        
        outputs['t_stats'] = os.path.abspath('t_stats.nii.gz')
        outputs['z_stats'] = os.path.abspath('z_stats.nii.gz')                                             

        return outputs
