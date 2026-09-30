import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from LAMP.diagnostic import Diagnostic
from LAMP.utils.xrays.calc_bragg_dispersion import calc_bragg_dispersion

class XrayCrystalSpec(Diagnostic):
    """
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''
    data_type = 'image'

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def get_proc_shot(self, shot_dict, calib_id=None, apply_disp=True, debug=False):
        """Return a processed shot using saved or passed calibrations.
        Wraps base diagnostic class, adding dispersion
        """
        # use diagnostic base function
        # loads calib id and run_img_calib for standard calibration routines
        img, x, y = super().get_proc_shot(shot_dict, calib_id=calib_id, debug=debug)

        # Apply dispersion?
        if (not hasattr(self, 'eV')) and ('detector' in self.calib_dict) and ('crystal' in self.calib_dict):
            self.set_dispersion()
        if hasattr(self, 'eV'):
            if 'dispersion_axis' in self.calib_dict and self.calib_dict['dispersion_axis'].lower() == 'y':
                y = self.eV
            else: # default to X axis
                x = self.eV

        # caluclate photon numbers?
        # this will be photons/sphere/eV ???
        # that way a mean lineout can be taken across spatial axis? (NOT a sum)
        if 'crystal' in self.calib_dict and 'int_refl' in self.calib_dict['crystal']:
            # non-dispersive angle of acceptance for a pixel
            theta_W = np.arctan(self.calib_dict['detector']['pixel_size'] / self.calib_dict['detector']['dist_source']) 

            # counts to electrons = * CCD_AD (the analogue-digitial conversion)
            # electrons to energy = * CCD_EH (the energy required to free an electron)  
            # energy to number of photons = / photon_energy
            # account for probability of absorption in CCD = / CCD_QE (Quantum efficiency)
            # account for transmission filtering = / transmission
            # account for crystal reflectivity and angular selection in dispersive axis = / integrated reflectivity
            # account for angular colelction in non-dispersive axis = / theta_W (angle subtended by a single pixel width)
            # convert to per sphere = * 4*pi
            # converting to per eV = / photon energy bin width
            bin_widths = np.abs(np.gradient(self.eV))

            # still working on this....
            img = ((img * 4 * np.pi * self.calib_dict['detector']['CCD_EH'] * self.calib_dict['detector']['CCD_AD']) / 
                        ((self.calib_dict['crystal']['int_refl']/1000) * self.calib_dict['detector']['CCD_QE'] * self.calib_dict['filters']['transmission'] * theta_W * self.eV * bin_widths))
                
            self.units = 'photons/sphere/eV' # ??

        return img, x, y

    def set_dispersion(self, pixels=None, central_eV=None, bragg_angle=None, dist_src_det=None, crystal_2d=None):
        """Convert pixels to eV using basic geometry calc.

        Currently using the diagnostic libray function calc_bragg_dispersion(...)

        Args:
            pixels: Array of pixel distances from centre (m)
            dist_src_det: Distance from source to dtector (m)
            crystal_2d: 2D lattice spacing of the crystal used (m)
            central_eV: Photon energy (eV) of zero pixel distance. Set this or bragg angle
            bragg_angle: Bragg angle of crystal (degrees)

        Returns:
            Array of dispersion (eV) across the detector pixels.
        """
        # If vars not passed, use calibration dict
        if not dist_src_det:
            dist_src_det = self.calib_dict['detector']['dist_source']
        if not crystal_2d:
            crystal_2d = self.calib_dict['crystal']['twod']
        if not central_eV:
            if 'central_eV' in self.calib_dict:
                central_eV = self.calib_dict['central_eV']
            else:
                if not bragg_angle:
                    if 'bragg_angle' in self.calib_dict['crystal']:
                        bragg_angle = self.calib_dict['crystal']['bragg_angle']
                    else:
                        print('Error, set_dispersion(); no central_eV or bragg_angle set')
                # calculate central eV from bragg angle
                central_eV = 1239.84 / ((crystal_2d*np.sin(np.deg2rad(bragg_angle)))*1e9)
        if not pixels:
            if 'num_pixels' in self.calib_dict['detector']:
                pixels = (np.arange(self.calib_dict['detector']['num_pixels']) - (self.calib_dict['detector']['num_pixels']/2)) * self.calib_dict['detector']['pixel_size']
            else:
                print('Error, set_dispersion(); no pixel distances passed or num_pixels for detector set')
        # Use library function
        self.eV = calc_bragg_dispersion(central_eV, crystal_2d, dist_src_det, pixels)
        return self.eV

    def feed_dispersion(self, eV):
        """Feed a pre-calculated detector spectral dispersion into XASA object.

        Note:
            Useful for comparing across objects with a known fixed dispersion.
            MUST be done before taking ROI or normalisations (for nabs_eV)
        Args:
            eV: Array of dispersion (eV) across the detector pixels.
        """
        self.eV = eV
        return
    
    def get_spectra(self, shot_dict, eV_min=None, eV_max=None, debug=False):

        img, x, y = self.get_proc_shot(shot_dict, debug=debug)

        if debug:
            plt.figure()
            plt.imshow(img)
            plt.title('Image for get_spectra()')
            plt.show(block=False)

        # untested
        if 'dispersion_axis' in self.calib_dict and self.calib_dict['dispersion_axis'].lower() == 'y':
            lineout = np.mean(img,1)
        else:
            lineout = np.mean(img,0)

        if hasattr(self, 'eV'):
            eV = self.eV
            if eV_min:
                lineout = lineout[eV > eV_min]
                eV = eV[eV > eV_min]
            if eV_max:
                lineout = lineout[eV < eV_max]
                eV = eV[eV < eV_max]
            return eV, lineout    
        else:
            return lineout

    def plot_histogram(self, timeframe, bins=100):

        shot_dicts = self.DAQ.get_shot_dicts(self.diag_name, timeframe)

        raw_data = []
        for shot_dict in shot_dicts:
            print(shot_dict)
            raw_data.append(self.get_shot_data(shot_dict))

        # plt.figure()
        # plt.imshow(raw_img)
        # plt.show(block=False)

        #print(len(np.array(raw_data).flatten()))

        fig = plt.figure()
        n, bins, patches = plt.hist(np.array(raw_data).flatten(), bins=bins, log=True)
        plt.show(block=False)

        return n, bins

    def get_hist_sig(self, shot_dict, lefti, righti, bin_edges = range(0,5000), view=False):
        """Written on the fly for experiment, definitely needs cleaned up!"""

        raw_data = self.get_shot_data(shot_dict)

        n, bins = np.histogram(np.array(raw_data).flatten(), bins=bin_edges)

        x = (bins[1:] + bins[:-1])/2
        fitn = np.concatenate((n[lefti[0]:lefti[1]], n[righti[0]:righti[1]]))
        fitx = np.concatenate((x[lefti[0]:lefti[1]], x[righti[0]:righti[1]]))
        z = np.polyfit(fitx, fitn, 3)
        p = np.poly1d(z)
        sig = n[lefti[1]+1:righti[0]] - p(x[lefti[1]+1:righti[0]])

        #print(np.sum(sig))

        if view:
            plt.figure()
            plt.semilogy(x,n)
            plt.semilogy(fitx,fitn)
            plt.semilogy(x[lefti[1]+1:righti[0]],p(x[lefti[1]+1:righti[0]]))
            plt.show(block=False)
            
            plt.figure()
            plt.plot(x[lefti[1]+1:righti[0]],sig)
            plt.ylabel('No. photons per ADU')
            plt.xlabel('ADUs')
            plt.show(block=False)

        return np.sum(sig)
    
    def get_hist_sigs(self, timeframe, lefti, righti, bin_edges = range(0,5000)):

        shot_dicts = self.DAQ.get_shot_dicts(self.diag_name, timeframe)

        sigs = []
        for shot_dict in shot_dicts:
            print(shot_dict)
            sigs.append(self.get_hist_sig(shot_dict, lefti, righti, bin_edges = bin_edges))
        
        return sigs
