import inspect
try:
    from eve_smlm.Utils import utilsHelper
except ImportError:
    from Utils import utilsHelper
import numpy as np

# Required function __function_metadata__
# Should have an entry for every function in this file
def __function_metadata__():
    return {
        "DriftCorr_AIM": {
            "required_kwargs": [
                {"name": "number_bins", "description": "Number of temporal bins used for drift-correction. Typical ~ 2000","default":2000,"type":int,"display_text":"Number of bins used in AIM"},
                {"name": "visualisation", "description": "Visualisation of the drift traces (Boolean).","default":True,"display_text":"Visualisation"},
            ],
            "optional_kwargs": [
            ],
            "help_string": "Corrects drift based on adaptive intersection maximization. See, and please cite Ma et al., ScienceAdvances, 2024.",
            "display_name": "Drift correction by AIM"
        },
    }

#-------------------------------------------------------------------------------------------------------------------------------
#Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

def DriftCorr_AIM(resultArray,findingResult,settings,**kwargs):
    """ 
    Implementation of AIM drift correction based on Ma et al. 2024 (https://www.science.org/doi/10.1126/sciadv.adm7765). 
    Implementation inspired by Picasso's Py-AIM implementation: https://github.com/jungmannlab/picasso/blob/master/picasso/aim.py
    """
    #Check if we have the required kwargs
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(__function_metadata__(),inspect.currentframe().f_code.co_name,kwargs) #type:ignore

    #Import the correct package
    from .aim import aim
    
    pxSize = float(settings['PixelSize_nm']['value'])
    
    time_prec_us = 1
    
    #Set user variables
    nr_bins = int(kwargs['number_bins'])
    visualisation=utilsHelper.strtobool(kwargs['visualisation'])
    
    resultArray=resultArray.dropna()
    
    # timevals should start at 1, and be integers
    timevals = np.floor((resultArray['t'].values + 1 - np.min(resultArray['t'].values)) / time_prec_us).astype(int)

    #time interval...
    segmentation = int(np.ceil(np.max(timevals) / nr_bins ))
    # find the segmentation bounds (temporal intervals)
    seg_bounds = np.concatenate((
        np.arange(0, np.max(timevals), segmentation), [np.max(timevals)]
    ))

    # # get the reference localizations (first interval)
    ref_x = resultArray['x'].values[timevals <= segmentation]/pxSize
    ref_y = resultArray['y'].values[timevals <= segmentation]/pxSize

    intersect_d = 4 #intersect distance in cam pixels
    roi_r = 1 #Radius of the local search region in camera pixels. Should be 
        #larger than the  maximum expected drift within segmentation.
    im_width = (np.max(resultArray['x'].values)-np.min(resultArray['x'].values))/pxSize
    
    ### RUN AIM TWICE ###
    # the first run is with the first interval as reference
    x_pdc, y_pdc, drift_x1, drift_y1 = aim.intersection_max(
        resultArray['x']/pxSize, resultArray['y']/pxSize, ref_x, ref_y,
        timevals, seg_bounds, intersect_d, roi_r, im_width, 
        aim_round=1, progress=None,
    )
    # # the second run is with the entire dataset as reference
    x_pdc, y_pdc, drift_x2, drift_y2 = aim.intersection_max(
        resultArray['x']/pxSize, resultArray['y']/pxSize, x_pdc, y_pdc,
        timevals, seg_bounds, intersect_d, roi_r, im_width, 
        aim_round=1, progress=None,
    )

    # add the drifts together from the two rounds and back to original units
    drift_x = (drift_x1 + drift_x2)*pxSize
    drift_y = (drift_y1 + drift_y2)*pxSize

    # shift the drifts by the mean value
    shift_x = np.mean(drift_x)
    shift_y = np.mean(drift_y)
    drift_x -= shift_x
    drift_y -= shift_y
    x_pdc += shift_x/pxSize  # x_pdc is in pixels, the drift in nm
    y_pdc += shift_y/pxSize
    
    import copy
    #Correct the resultarray for the drift
    drift_corr_locs = copy.deepcopy(resultArray)
    drift_corr_locs.loc[:,'x'] = x_pdc.values*pxSize
    drift_corr_locs.loc[:,'y'] = y_pdc.values*pxSize

    if visualisation:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, len(drift_x) + 1), drift_x,label='Drift in X')
        plt.plot(range(1, len(drift_y) + 1), drift_y,label='Drift in Y')
        plt.xlabel("Time (us)")
        plt.ylabel("Drift (px)")
        plt.legend()  # Added legend
        plt.show()
    
    
    performance_metadata = f"Driftcorrection AIM applied with settings {kwargs}."
    
    return drift_corr_locs, performance_metadata