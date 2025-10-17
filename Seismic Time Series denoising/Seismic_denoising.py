#!/usr/bin/env python3
import h5py
import numpy as np
from scipy.linalg import eigh
import pywt #python wavelet functions
import matplotlib.pyplot as plt


class UrbanSeismicDenoiser:
    
    #-------- load and filter for low SNR data-------------------
    #----------------------------------------------------------------------
    def load_data(self, input_file, max_samples=50):
        
        with h5py.File(input_file, 'r') as f: #i had no idea how to handle h5py files
            X = f['X'][:]
            Y = f['Y'][:]
            mag = f['mag'][:]
            sncls = f['sncls'][:]
            snr = f['snr'][:]
            dist = f['dist'][:]
            
        #get CI network(s)
        networks = [s.decode('utf-8').split('.')[0] for s in sncls]
        #filter for CI network for  small Earthquakes, local (small dist from sensor), bottom 20% SNR
        #I am using the 'masks' apparently this is a quicker than for loops, I had run time issues
        ci_mask = np.array(networks) == 'CI'
        small_eq_mask = mag <= 3.5
        local_mask = dist <= 50
        known_mask = Y != 2
        low_snr_mask = snr <= np.percentile(snr, 20)
        
        #meeting all reqs
        combined_mask = ci_mask & small_eq_mask & local_mask & known_mask & low_snr_mask
        indices = np.where(combined_mask)[0][:max_samples]
        
        self.X = X[indices]
        self.Y = Y[indices]
        self.snr = snr[indices]
        
        print(f"Loaded {len(self.X)} samples, Mean SNR: {np.mean(self.snr):.1f}±{np.std(self.snr):.1f}")
        return self.X, self.Y, self.snr
    
    #--------Random Matrix Theory denoising (eigendecomposition) :) -----------
    #-----------------------------------------------------------------------------
    def denoise_rmt(self, waveforms, n_components=5):
        data_matrix = np.array(waveforms)
            
        data_centered = data_matrix - np.mean(data_matrix, axis=1, keepdims=True) #center data
        correlation = np.cov(data_centered) #correlation matrix
        eigenvals, eigenvecs = eigh(correlation) #eigen decomp
        eigenvals = eigenvals[::-1]
        eigenvecs = eigenvecs[:, ::-1] #reorder in terms of eigenvalues
        
        q = data_matrix.shape[0] / data_matrix.shape[1] #aspect ration of data set
        lambda_max = (1 + np.sqrt(q))**2 #Marchenko-Pastur threshold
        signal_components = eigenvals > lambda_max
        n_signal = max(n_components, np.sum(signal_components)) #boolean mask
        
        signal_eigenvals = eigenvals[:n_signal] #Keep only the first n eigenvalues (the strongest signal components)
        signal_eigenvecs = eigenvecs[:, :n_signal] #same for vecotrs
        projection = signal_eigenvecs @ np.diag(signal_eigenvals) @ signal_eigenvecs.T #rebuild matrix
        denoised = projection @ data_centered #combine with orginal
        
        return denoised
    
    #-------- wavelet denoising (soft thresholding) ---------------
    #-----------------------------------------------------------------------------
    def denoise_wavelet(self, waveform, wavelet='db4', level=4): #decomposed into 4 levels/freq scales
        coeffs = pywt.wavedec(waveform, wavelet, level=level)
        
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 #0.6745 const for normal distrib
        threshold = sigma * np.sqrt(2 * np.log(len(waveform))) #Universal threshold for signal vs noise
        
        coeffs_thresholded = [coeffs[0]] #Keep the low-frequency component untouched
        for i in range(1, len(coeffs)):
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='soft')) #predefined func 
        #If coefficient < threshold: set to zero (remove noise)
        denoised = pywt.waverec(coeffs_thresholded, wavelet)
        
        if len(denoised) > len(waveform):
            denoised = denoised[:len(waveform)] #making sure lenghts match by trimming, had some issues here
        elif len(denoised) < len(waveform):
            denoised = np.pad(denoised, (0, len(waveform) - len(denoised)), mode='edge')
            
        return denoised
    
    #-------- RMT + Wavelet denoising --------------------
    #------------------------------------------------------------------------
    def denoise_combined(self, waveforms):
        rmt_denoised = self.denoise_rmt(waveforms)
        
        combined_denoised = []
        for wave in rmt_denoised:
            wavelet_denoised = self.denoise_wavelet(wave)
            combined_denoised.append(wavelet_denoised)
        
        return np.array(combined_denoised)
    
    #--------Get lowest SNR data ---------------------------
    #----------------------------------------------------------
    def get_lowest_snr_samples(self, input_file, n_samples=2, skip =0):
        with h5py.File(input_file, 'r') as f:
            X_all = f['X'][:]
            Y_all = f['Y'][:]
            snr_all = f['snr'][:]
            sncls_all = f['sncls'][:]
            dist_all = f['dist'][:]
        
        networks_all = np.array([s.decode('utf-8').split('.')[0] for s in sncls_all])
        #bottom 2% signal noise ratios
        noisy_mask = (networks_all == 'CI') & (snr_all <= np.percentile(snr_all, 2)) & (dist_all <= 50) & (Y_all != 2)
        noisy_idx = np.where(noisy_mask)[0][skip:skip+n_samples]
        
        return X_all[noisy_idx], Y_all[noisy_idx], snr_all[noisy_idx]
    
    #-------- comparison of noisy, RMT only, RMT + wavelet ------------------
    #-----------------------------------------------------------------------------
    def plot_denoising_comparison(self, waveforms, labels, snr_values):
        rmt_denoised = self.denoise_rmt(waveforms)
        combined_denoised = self.denoise_combined(waveforms)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        
        for i in range(len(waveforms)):
            row = i
            time = np.linspace(0, 6, len(waveforms[i]))
            
            ax_noisy = axes[row, 0]
            ax_noisy.plot(time, waveforms[i], 'k-', linewidth=0.8)
            ax_noisy.set_title(f"Original Noisy\nSNR: {snr_values[i]:.2f}", 
                              fontsize=10)
            ax_noisy.set_xlabel('Time')
            ax_noisy.set_ylabel('Amplitude')
            ax_noisy.grid(True, alpha=0.3)
            
            
            ax_rmt = axes[row, 1]
            ax_rmt.plot(time, rmt_denoised[i], 'b-', linewidth=0.8)
            ax_rmt.set_title(f"RMT Denoised", fontsize=10)
            ax_rmt.set_xlabel('Time')
            ax_rmt.set_ylabel('Amplitude')
            ax_rmt.grid(True, alpha=0.3)
            
            
            ax_combined = axes[row, 2]
            ax_combined.plot(time, combined_denoised[i], 'g-', linewidth=0.8)
            ax_combined.set_title(f"RMT + Wavelet", 
                                 fontsize=10)
            ax_combined.set_xlabel('Time')
            ax_combined.set_ylabel('Amplitude')
            ax_combined.grid(True, alpha=0.3)
            
        
        plt.tight_layout()
        plt.show()
        


#--------Main  -----------------------
#-----------------------------------------------------------------------------
def main():
    denoiser = UrbanSeismicDenoiser()
    
    #load low-SNR data
    X, Y, snr = denoiser.load_data('scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5') #this is the data file from the California Sismic center website
    
    low_snr_waves, low_snr_labels, low_snr_vals = denoiser.get_lowest_snr_samples(
        'scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5', n_samples=2, skip =2)
    
    print("\n RMT + Wavelet denoising on selected noisey samples from Urban California Areas")
    denoiser.plot_denoising_comparison(low_snr_waves, low_snr_labels, low_snr_vals)
    
    return denoiser


if __name__ == "__main__":
    denoiser = main()