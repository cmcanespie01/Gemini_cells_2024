import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
from LAMP.diagnostic import Diagnostic
from LAMP.utils.plotting import plot_montage
from skimage.measure import label, regionprops
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

class Timepix(Diagnostic):
    """Timepix3
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''
    data_type = 'text'
    data_options = {'skip_header': True} # should default to this, but overwrite if in diagnostics.toml?

    # TODO
    # - histogram the freq. of each pixel position hit (for identifying breakages)
    #   Previously have removed pixels that have too many hits (something dodgy with them)
    # - Would a dataframe of all hits speed any of this up? (perhaps for after beamtime)
    # - marked output? (when cluster types identified)

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def make_images(self, shot_dict, saving=False):
        """Build image arrays from raw TimePix data"""

        raw_data = self.get_shot_data(shot_dict)

        # no data?
        if len(raw_data) < 1:
            return np.zeros((256,256)),np.zeros((256,256))
        # single hit fix
        if len(np.shape(raw_data)) == 1: 
            raw_data = np.reshape(raw_data,(1,len(raw_data)))
        # katherine readout? add two dummy columns, and switch other ToTs and FToAs
        if np.shape(raw_data)[1] == 4:
            orig_data = np.copy(raw_data)
            raw_data = np.zeros((np.shape(raw_data)[0],6))
            raw_data[:,1] = orig_data[:,0]
            raw_data[:,2] = orig_data[:,1]
            raw_data[:,3] = orig_data[:,3]
            raw_data[:,4] = orig_data[:,2]

        # Index | Matrix Index | ToA | ToT | FToA | Overflow
        # ToA = time of arrival
        # ToT = time over the threshold (ToT corresponds to energy of particle)
        # FToA is used in conjunction with ToA to give a more accurate timestamp
        pixels = raw_data[:,1].astype(int)
        ToAs = raw_data[:,2].astype(int)
        ToTs = raw_data[:,3].astype(int)
        FToAs = raw_data[:,4].astype(int)
        #overflows = raw_data[:,5].astype(int)

        # Matrix Index - position (index) of the pixel in pixel matrix - 
        # for single detector 0 - 65535, where 0 - 255 are pixels in first row, 
        # 256 - 511 second row, etc. x=matrixIndex mod 256, y = matrixIndex / 256.
        # rows = np.floor(pixels/256).astype(int)
        # cols = np.mod(pixels,256).astype(int)
        cols = np.floor(pixels/256).astype(int)
        rows = np.mod(pixels,256).astype(int)
        

        # make ToT image
        # Does this work? what about overlapping pixels with different timestamps? how to handle that?
        # This will be more like an integrated hitmap
        ToT_img = np.zeros((256,256))
        ToT_img[cols,rows] = ToTs

        # Make ToA image
        cToA_img = np.zeros((256,256))
        # cToA (ns) = (ToA * 25) - (FToA * (25/16))
        cToAs = (ToAs * 25) - (FToAs * (25/16))
        cToA_img[cols,rows] = cToAs

        # Check unique timestamps?
        # Round timestamps and integrated within one second?

        # does it ever hit some limit? there is a note from 2018 MATLAB code:
        #% reset cToAs... 32bit can only support 2 seconds! (2^32=4.3e9 ... /2 for signed)
        #% ADD an offset to differenetiate between no hits...
        # I think it means an early hit causes issue and you have remove/reset...

        # save images?

        return ToT_img, cToA_img

    def get_filtered_ToT(self, shot_dict, filters_lower={}, filters_upper={}):

        ToT_img, cToA_img = self.make_images(shot_dict)

        clusters = self.get_clusters(shot_dict,filters_lower=filters_lower, filters_upper=filters_upper)

        filtered_image = np.zeros(np.shape(ToT_img))

        for index, cluster in clusters.iterrows():
            for pixels in cluster['coords']:
                filtered_image[pixels[0],pixels[1]] = ToT_img[pixels[0],pixels[1]]
                
        return filtered_image

    def integrate_ToT(self, timeframe, filters_lower=None, filters_upper=None):
        # stack ToT, images over series of shots

        return

    def plot_images(self, shot_dict):
        ToT_img, cToA_img = self.make_images(shot_dict)

        fig, axs = plt.subplots(1, 2)

        im = axs[0].imshow(ToT_img, cmap=self.colormap())
        axs[0].set_title('ToT')
        cb = plt.colorbar(im, ax=axs[0])

        im = axs[1].imshow(cToA_img, cmap=self.colormap())
        axs[1].set_title('ToA+FToA')
        cb = plt.colorbar(im, ax=axs[1])

        fig.suptitle(f"{self.config['name']}: {shot_dict}")

        return fig

    def get_clusters(self, shot_dict, threshold=1, connectivity=2, min_pixels=1, filters_lower={}, filters_upper={}):
        """
        Identify contiguous clusters above a threshold in a 2D image.

        Parameters
        ----------
        image : ndarray
            2D image array.
        threshold : float
            Signal threshold.
        connectivity : int
            1 = edge-connected (4-neighbour)
            2 = edge+corner connected (8-neighbour)
        min_pixels : int
            Minimum cluster size to retain.

        Returns
        -------
        DataFrame
            One row per cluster.
        """

        ToT_img, cToA_img = self.make_images(shot_dict)
        image = ToT_img

        # Binary mask
        mask = image > threshold

        # Label connected regions
        labels = label(mask, connectivity=connectivity)

        results = []

        # are labels to clusters themselves?

        for region in regionprops(labels, intensity_image=image):

            if region.area < min_pixels:
                continue

            # Bounding box
            min_row, min_col, max_row, max_col = region.bbox

            width = max_col - min_col
            height = max_row - min_row

            # Intensity statistics
            total_counts = region.image_intensity[region.image].sum()
            mean_counts = region.image_intensity[region.image].mean()
            max_counts = region.image_intensity[region.image].max()

            # Shape properties
            major_axis = region.axis_major_length
            minor_axis = region.axis_minor_length

            aspect_ratio = (
                major_axis / minor_axis
                if minor_axis > 0 else np.inf
            )

            results.append({
                # is this the cluster?
                "label": region.label,

                # list of pixels??
                'coords': region.coords,

                # Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
                'angle': region.orientation,

                # Position
                "centroid_x": region.centroid_weighted[1],
                "centroid_y": region.centroid_weighted[0],

                # Pixel count
                "num_pixels": region.area,

                # Bounding box
                "width": width,
                "height": height,

                # Intensity
                "total_counts": total_counts,
                "mean_counts": mean_counts,
                "max_counts": max_counts,

                # Shape
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "aspect_ratio": aspect_ratio,
                "eccentricity": region.eccentricity, # Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
                #"solidity": region.solidity,
                "extent": region.extent, # Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
                #"orientation_rad": region.orientation,

                # Shape moments
                "equivalent_diameter":
                    region.equivalent_diameter_area,
                "perimeter":
                    region.perimeter,

                # "slice" (tuple of slices)
                # A slice to extract the object from the source image.

                #"inertia_tensor"
                #Inertia tensor of the region for the rotation around its mass.
            })

        # centroid_x, centroid_y	Intensity-weighted centre
        # num_pixels	Number of pixels in cluster
        # width, height	Bounding box dimensions
        # total_counts	Integrated signal
        # major_axis, minor_axis	Ellipse fit dimensions
        # aspect_ratio	Elongation
        # eccentricity	0=circular, 1=line-like
        # solidity	Area / convex hull area
        # extent	Area / bounding-box area
        # orientation_rad	Principal axis angle
        # equivalent_diameter	Diameter of equal-area circle
        # perimeter	Cluster boundary length

        # For scientific imaging (CCD, X-ray, plasma diagnostics, etc.), consider adding:
        # region.moments_central
        # region.inertia_tensor
        # region.inertia_tensor_eigvals
        # from which you can derive RMS widths, covariance matrices, principal axes, and beam-emittance-style characterisations that are often more useful than simple bounding-box widths.

        if len(results) < 1:
            return pd.DataFrame([])

        # now turn into a DataFrame
        df = pd.DataFrame(results)
        # and apply filters
        mask = pd.Series(True, index=df.index)
        for filters_lower_key, filters_lower_val in filters_lower.items():
            mask &= df[filters_lower_key] >= filters_lower_val
        for filters_upper_key, filters_upper_val in filters_upper.items():
            mask &= df[filters_upper_key] <= filters_upper_val
        filtered_df = df[mask]
        # if min_width is not None:
        #     mask &= df['width'] >= min_width
        # filtered_df = df[
        #     (df['width'] >= 10) &
        #     (df['height'] <= 20) &
        #     (df['pixels'] >= 150)
        # ]

        # alternatively, something like;
        # filtered_df = df.query(
        #     "width >= 10 and height <= 20 and pixels >= 150"
        # )

        return filtered_df
    
    #def montage(self,timeframe):

    def plot_summary(self,timeframe, shotnums=None, filters_lower={}, filters_upper={}, montage=True):

        # get all shots, for whatever form is passed, and loop through each shot dict
        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe)
        num_raw_hits = []
        num_filt_hits = []
        #shot_labels = []

        all_clusters = {}

        for shot_dict in shot_dicts:

            if shotnums is not None:
                if shot_dict['shotnum'] not in shotnums:
                    continue

            print(shot_dict)

            # # dodgy massive file?
            # filepath = self.DAQ.get_filepath(self.config['name'],shot_dict)
            # if os.path.getsize(filepath) > 1000000:
            #     print(f'Skipping (file too large): {shot_dict}')
            #     continue
            
            # make images, raw and filtered
            ToT_img, cToA_img = self.make_images(shot_dict)
            if 'ToT_imgs' in locals():
                ToT_imgs = np.concatenate((ToT_imgs, np.atleast_3d(ToT_img)), axis=2)
            else:
                ToT_imgs = np.atleast_3d(ToT_img)
            if 'ToT_img_all' in locals():
                ToT_img_all = ToT_img_all + ToT_img
            else:
                ToT_img_all = ToT_img.copy()
            ToT_img_filtered = self.get_filtered_ToT(shot_dict, filters_lower=filters_lower,filters_upper=filters_upper)
            if 'ToT_img_filtered_all' in locals():
                ToT_img_filtered_all = ToT_img_filtered_all + ToT_img_filtered
            else:
                ToT_img_filtered_all = ToT_img_filtered.copy()

            clusters = self.get_clusters(shot_dict)
            clusters_filt = self.get_clusters(shot_dict, filters_lower=filters_lower,filters_upper=filters_upper)

            num_raw_hits.append(len(clusters))
            num_filt_hits.append(len(clusters_filt))

            if len(clusters) > 0:

                # what other stats? average size/counts/width/height/major axis? / angle
                # this is all for unfiltered
                
                for key in clusters.columns.values.tolist():
                    if key in all_clusters:
                        all_clusters[key].extend(clusters[key].to_list())
                    else:
                        all_clusters[key] = clusters[key].to_list()

                num_pixels = clusters['num_pixels'].to_list()
                angles = clusters['angle'].to_list()
                widths = clusters['width'].to_list()
                heights = clusters['height'].to_list()
                counts  = clusters['total_counts'].to_list()

                mean_num_pixels = np.mean(num_pixels)

        # plot integrated ToT maps, filtered and not
        fig, axs = plt.subplots(1, 2)
        im = axs[0].imshow(ToT_img_all, cmap=self.colormap())
        axs[0].set_title('ToT')
        cb = plt.colorbar(im, ax=axs[0])
        im = axs[1].imshow(ToT_img_filtered_all, cmap=self.colormap())
        axs[1].set_title('ToT Filtered')
        cb = plt.colorbar(im, ax=axs[1])
        fig.suptitle(f"{self.config['name']} - Intregrated shots ({timeframe})")
        
        # plot histograms of cluster properties
        # Reminder of properties:'label', 'coords', 'angle', 'centroid_x', 'centroid_y', 'num_pixels',
        # 'width', 'height', 'total_counts', 'mean_counts', 'max_counts',
        # 'major_axis', 'minor_axis', 'aspect_ratio', 'eccentricity', 'extent',
        # 'equivalent_diameter', 'perimeter'
        
        fig, axs = plt.subplots(1, 4)
        axs[0].hist(all_clusters['width'], bins=range(50))
        axs[0].set_title('Width')
        axs[0].set_xlabel('Pixels')
        axs[0].set_ylabel('Frequency')
        axs[1].hist(all_clusters['height'], bins=range(50))
        axs[1].set_title('Height')
        axs[1].set_xlabel('Pixels')
        # axs[2].hist(all_clusters['angle'])
        # axs[2].set_title('Angle')
        # axs[2].set_xlabel('?')
        axs[2].hist(all_clusters['total_counts'], bins=range(2500))
        axs[2].set_title('Total Counts')
        axs[2].set_xlabel('Counts')
        axs[3].hist(all_clusters['mean_counts'], bins=range(100))
        axs[3].set_title('Mean Counts')
        axs[3].set_xlabel('Counts')
        plt.suptitle(f'{self.config['name']} - All clusters: {timeframe}')
        plt.show(block=False)
        
        # montages?
        if montage:
            num_per_row = 8
            fig, ax = plot_montage(ToT_imgs, transpose=True, vmin=0, num_rows=int(np.ceil(len(shot_dicts)/num_per_row)), colormap='electron_beam', colormap_option=0.5) # colormap bodge - should be able to pass a colormap
            ax.axis('equal')

        # plot stats
        plt.figure()
        plt.plot(num_raw_hits, label='Raw')
        plt.plot(num_filt_hits, label='Filtered')
        plt.title('Number of clusters (hits)')
        plt.ylabel('')
        #plt.xlabel(shot_labels)
        plt.legend()
        plt.show(block=False)

        results = {}
        results['mean_num_hits'] = np.mean(num_raw_hits)
        results['std_num_hits'] = np.std(num_raw_hits)
        results['mean_num_hits_filtered'] = np.mean(num_filt_hits)
        results['std_num_hits_filtered'] = np.std(num_filt_hits)

        return results
    
    def colormap(self):
        # modify the viridis colormap, so that the top color is a green (better visible on the H&E pink), and such that
        # the value 0 leads to a transparent color
        #cm_orig = cm.get_cmap("viridis", 256) # winter? viridis?
        cm_orig = colormaps['viridis']
        colors = cm_orig(np.linspace(0, 1, 256))
        # set the color of zero to be white
        colors[0, :] = [1.0, 1.0, 1.0, 1.0]
        return LinearSegmentedColormap.from_list("truncated_viridis", colors)  