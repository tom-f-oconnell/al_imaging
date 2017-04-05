
# for backwards compability with Python 2.7
from __future__ import print_function
from __future__ import division

from . import plotting as tplt
from . import odors
from . import util as u
from . import checks
from . import percache_helper

import numpy as np
import pandas as pd

# TODO how much of their use is redundant?
from os.path import join, split, isfile
from os import listdir
# TODO replace with cache decorators and offload remainder into util
import pickle

import tifffile
import thunder as td
from registration import CrossCorr
from registration.model import RegistrationModel
import cv2
from PIL import Image, ImageDraw

import percache
# TODO conflict with other caches of same name?
cache = percache.Cache('.cache')
cache = checks.check_args_hashable(cache)

# TODO either cache w/ decorator or save as another TIF
@cache
def generate_motion_correction(directory, images, use_cache=True):
    """
    Aligns images within a movie of one plane to each other. Saves the transform.
    """
    print('starting thunder registration...')
    print(directory)

    # TODO if use_cache:
    # registering to the middle of the stack, to try and maximize chance of a good registration
    middle = round(images.shape[0] / 2.0)

    # TODO implement w/ consistent interface
    if type(images) is not np.ndarray:
        reference = images[middle,:,:].toarray().squeeze()
    else:
        reference = images[middle,:,:].squeeze()

    reg = CrossCorr()
    model = reg.fit(images, reference=reference)

    print('done')
    return model


def correct_and_write_tif(directory, transforms=None):
    raise NotImplementedError('not using OME-TIFF input anymore')
    '''
    # TODO load model if it is None
    print('applying transformation...')

    files = frame_filenames(directory)
    first = load_frame(files[0])

    #registered = np.zeros((*first.shape, len(files)))
    registered = np.empty(( *first.shape, len(files))) * np.nan
    # check no nan?

    for i, f in enumerate(files):
        print(str(i) + '/' + str(len(files)), end='        \r')
        frame = load_frame(f)
        # TODO add support for making RegistrationModels from matrices
        #registered[:,:,i] = RegistrationModel({(0,): transforms[i,:]}).transform(frame)
        model = transforms[f]
        registered[:,:,i] = model.transform(frame).toarray()

    # TODO deal with metadata too?
    # TODO test this is readable and is identical to data before writing
    reg_file_out = split(directory)[-1] + '_registered.bf.tif'
    path_out = join(directory, reg_file_out)
    print('saving registered TIF to', path_out, '...', end='')

    normalized = cv2.normalize(registered, None, 0.0, 2**16 - 1, cv2.NORM_MINMAX).astype(np.uint16)

    expanded = np.expand_dims(normalized, 0)
    # this empirically seems to be correct, but apparently goes against
    # online documentation
    expanded = np.swapaxes(expanded, 2, 3)
    expanded = np.swapaxes(expanded, 1, 2)
    tifffile.imsave(path_out, expanded, imagej=True)

    print('done.')
    '''


def get_active_region(deltaF, thresh, invert=False, debug=False):
    """
    Returns the largest contour in the delta image.  This should enable glomerulus identification.

    Args:
        deltaF (np.ndarray): uint8 encoded image (all pixels from 0 to 255)
    """

    if not invert:
        thresh_mode = cv2.THRESH_BINARY

    else:
        thresh_mode = cv2.THRESH_BINARY_INV

    #ret, threshd = cv2.threshold(deltaF, thresh, np.max(deltaF), thresh_mode)
    ret, threshd = cv2.threshold(deltaF, thresh, 255, thresh_mode)

    kernel = np.ones((3,3), np.uint8)
    first_expanded = cv2.dilate(threshd, kernel, iterations=1)
    eroded = cv2.erode(first_expanded, kernel, iterations=2)
    # this is largely to get smooth ROIs
    dilated = cv2.dilate(eroded, kernel, iterations=3)

    # it is important not to use an approximation. drawContours doesn't behave well,
    # and won't include all pixels in contour correctly later.
    #print(dilated.dtype)
    #print(dilated.shape)
    img, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:

        if debug:
            #tplt.hist_image(deltaF, title='Histogram of image without detectable contours')
            return None, threshd, dilated, None

        # TODO but we don't really expect to find all glomeruli for which we present the cognate
        # odors...
        else:
            raise RuntimeError('either the data is bad or get_active_region parameters are set ' + \
                'such that no contours are being found')

    areaArray = []
    # returns the biggest contour
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    #first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    #find the nth largest contour [n-1][1], in this case 2
    largestcontour = sorteddata[0][1]

    if debug:
        #print('area of largest contour=', sorteddata[0][0])

        # so that we can overlay colored information about contours for 
        # troubleshooting purposes
        # last axis should be color
        contour_img = np.ones((dilated.shape[0], dilated.shape[1], \
                3)).astype(np.uint8)

        # TODO 
        # why does the channel coming from dilated appear to be different intensities 
        # sometimes? isn't dilated binary? i guess it isn't... from looking at range
        contour_img[:,:,0] = dilated * 150

        #dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        # args: desination, contours, contour id (neg -> draw all), color, thickness
        cv2.drawContours(contour_img, largestcontour, -1, (0,0,255), 3)

        x, y, w, h = cv2.boundingRect(largestcontour)
        # args: destination image, point 1, point 2 (other corner), color, thickness
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0,255,0), 5)

        return largestcontour, threshd, dilated, contour_img

    else:
        return largestcontour


# TODO make work for imagej stuff too
def pixels_in_contour(img, contour):
    """
    Assumes that the dimensions of each plane are the last two dimensions.
    """
    # because squeeze doesn't work on Images...
    # i guess because it can't change # of images in sequence?
    if not type(img) is np.ndarray:
        img = img.toarray()

    img = np.squeeze(img)

    if len(img.shape) == 2:
        mask = np.zeros(img.shape, np.uint8)

        # TODO keep in mind that this is different from displayed contour, if using any
        # thickness to display
        cv2.drawContours(mask, contour, -1, 1, -1)
        return img[np.nonzero(mask)]

    else:
        # TODO test this more rigourously
        return np.array([pixels_in_contour(img[i,:,:], contour) for i in range(img.shape[0])])


# TODO move to a file in test directory
def test_pixels_in_contour():
    img = np.zeros((128,128)).astype(np.uint8)

    # generate an asymmetric test image
    s1 = 7
    e1 = 17
    s2 = 11
    e2 = 13
    # 0 < thresh < val
    val = 255
    img[s1:e1,s2:e2] = val

    thresh = 125
    thresh_mode = cv2.THRESH_BINARY
    ret, threshd = cv2.threshold(img, thresh, 255, thresh_mode)

    img, contours, hierarchy = cv2.findContours(threshd, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    # the function we actually want to test
    pixel_values = pixels_in_contour(img, contours[0])

    assert np.sum(pixel_values) == (e1 - s1) * (e2 - s2) * val, 'sum=' + str(np.sum(pixel_values))

    img[s1,s2] = 0

    pixel_values = pixels_in_contour(img, contours[0])
    assert np.sum(pixel_values) == (e1 - s1) * (e2 - s2) * val - val, 'sum wrong after modifying img'

    img = np.zeros((1050, 512, 512))
    print(pixels_in_contour(img, contours[0]))
    print(pixels_in_contour(img, contours[0]).shape)


def glomerulus_contour(img, debug=False):
    uint8_normalized_img = cv2.normalize(img, None, 0.0, 255.0, cv2.NORM_MINMAX)\
            .astype(np.uint8)
    thresh = np.percentile(uint8_normalized_img, 99)

    # if debug is false, it will just return hotspot
    contour, threshd, dilated, contour_img = \
            get_active_region(uint8_normalized_img, thresh=thresh, \
            debug=debug)

    return contour


def ijroi2mask(ijroi, shape):
    poly = [(p[1], p[0]) for p in ijroi]
    img = Image.new('L', shape, 0)
    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    return np.array(img)


def extract(img, contour):
    return np.mean(pixels_in_contour(img, contour), axis=1)


# TODO test
def extract_mask(img, mask):
    nzmask = np.nonzero(mask)
    return np.mean(img[:,nzmask[0],nzmask[1]], axis=1)


def xy_motion_score(series):
    """
    Return a score (lower better) correlated with amount of XY motion between frames.
    """
    # TODO test it never decreases for perturbations to an image with itself (+ noise?)
    assert False, 'not implemented'


@cache
def process_session(d, data_stores, params, recompute=False):
    """
    Analysis pipeline from directory name to the projected, averaged, delta F / F
    image, for one group of files (one block within one fly, not one fly).

    Args:
    data_stores: holds two dicts which this function adds more dictionaries with images to

    Returns:
    df: pandas.DataFrame that is either empty, or has extracted data from current fly / session
    
    """
    print(d)
    projections, rois = data_stores
    # TODO deal with recompute flag in Memorize / percache decorator

    # TODO factorize these out
    # we should have previously established the information loaded below
    # exists for this directory
    with open(join(d,'generated_pin2odor.p'), 'rb') as f:
        connections = pickle.load(f)
        pin2odor = dict(map(lambda x: (x[0], x[1]), connections))

    with open(join(d,'generated_stimorder.p'), 'rb') as f:
        pinsetlist = pickle.load(f)

    timg_meta = u.get_thorimage_metafile(d)
    original_fps = u.get_effective_framerate(timg_meta)

    tif_name = join(d, split(d)[-1] + '_ChanA.tif')
    movie = td.images.fromtif(tif_name)
    # this functions to squeeze out unit dimension and put number of images
    # in the correct position. kind of a hack.
    movie = td.images.fromarray(movie)

    ts_df = u.load_thor_hdf5(d, exp_type='imaging')

    averaging = u.get_thor_averaging(timg_meta)
    samprate_Hz = u.get_thorsync_samprate(u.get_thorsync_metafile(d))

    # careful... will have to crop movie same way each time for motion correction to be valid
    # check it by eye
    movie, _ = crop_trailing_frames(movie, ts_df, averaging, samprate_Hz)

    model = generate_motion_correction(d, movie, use_cache=(not recompute))
    # TODO break out and memorize? or just save TIFFs?
    corrected = model.transform(movie)

    # TODO handle better. figure out why this had KeyErrors previously
    # convert the list of pins (indexed by trial number) to a list of odors indexed the same
    try:
        print(pin2odor)
        print(pinsetlist)
        odorsetlist = [frozenset({pin2odor[p] for p in s}) for s in pinsetlist]

    except KeyError:
        return pd.DataFrame()

    stimulus_panel = set(odorsetlist)

    repeats = params['repeats']
    num_trials = len(pinsetlist)
    frames_per_trial = movie.shape[0] // num_trials

    ################# TODO remove me and get thunder working
    onset = params['recording_start_to_odor_s']
    actual_fps = u.get_effective_framerate(timg_meta)
    frames_before = int(np.floor(onset * actual_fps)) + 1
    def delta_fluorescence(signal, windows):
        deltaFs = []
        frames_before = int(np.floor(onset * actual_fps))
        for w in windows:
            region = signal[w[0]:w[1],:,:] + 0.001
            baseline_F = np.mean(region[:frames_before,:,:], axis=0)
            delta_F_normed = np.zeros(region.shape) * np.nan
            for i in range(delta_F_normed.shape[0]):
                delta_F_normed[i,:,:] = (region[i,:,:] - baseline_F) \
                    / baseline_F
            deltaFs.append(delta_F_normed)
        return deltaFs

    kernel_size = 3
    kernel = (kernel_size, kernel_size)
    # if this is zero, it is calculated from kernel_size, which is what i want
    sigmaX = 0

    for i in range(movie.shape[0]):
        movie[i,:,:] = cv2.GaussianBlur(movie[i,:,:], kernel, sigmaX)

    # neither toblocks nor reshape permit splitting images into different groups
    # fix that? against a design decision?
    # seems toblocks is more appropriate than reshaping
    # because they enforce constraints on dimensions in reshaping
    # TODO thunderize if possible. change thunder if OK.
    groups = (movie[i:i+frames_per_trial,:,:] for i in \
            range(round(movie.shape[0] / frames_per_trial)))

    windows = [(i*frames_per_trial,(i+1)*frames_per_trial) \
            for i in range(round(movie.shape[0] / frames_per_trial))]
    
    deltaF = delta_fluorescence(movie, windows)

    #################
    """
    # neither toblocks nor reshape permit splitting images into different groups
    # fix that? against a design decision?
    # seems toblocks is more appropriate than reshaping
    # because they enforce constraints on dimensions in reshaping
    # TODO thunderize if possible. change thunder if OK.
    groups = (corrected[i:i+frames_per_trial,:,:] for i in \
            range(round(corrected.shape[0] / frames_per_trial)))

    # might modify in place? unclear from docs
    filtered = (x.gaussian_filter(sigma=3) for x in groups)
    
    # still necessary?
    nonzero = [x.subtract(-0.001) for x in filtered]

    onset = params['recording_start_to_odor_s']
    actual_fps = get_effective_framerate(timg_meta)
    frames_before = int(np.floor(onset * actual_fps)) + 1
    # squeeze?
    prestimulus_baseline = [x[:frames_before,:,:].mean() for x in nonzero]
    # TODO could probably make more functional
    #prestimulus_baseline = (x[:frames_before,:,:].mean() for x in nonzero)

    #print('prestim baseline shape', prestimulus_baseline[0].shape)

    deltaF = [iseq.map(lambda i: i/base) for iseq, base in zip(nonzero, prestimulus_baseline)]
    """
    ####?
    # what about now?

    unique_stimuli = len(stimulus_panel)
    by_stimulus = [deltaF[i*repeats:(i+1)*repeats] for i in range(unique_stimuli)]

    # TODO try not to have these by numpy arrays yet
    # to take advantage of thunder
    #avgdF = [np.mean(np.array([x.toarray() for x in g]), axis=0) for g in by_stimulus]
    avgdF = [np.mean(np.array([x for x in g]), axis=0) for g in by_stimulus]

    # can't do this b/c they are numpy arrays not Images
    #maxdF = [i.max_projection(axis=0) for i in avgdF]
    maxdF = [np.max(i, axis=0) for i in avgdF]
    ####?

    stimperblock = odorsetlist[0::repeats]

    session_projections = dict(zip(stimperblock, maxdF))
    # holds images for debugging later
    session_rois = dict()

    '''
    for stim, proj in session_projections.items():
        if singular(stim):
            odor = tuple(stim)[0]

            if odors.is_private(odor):
                glom = odors.uniquely_activates[odor]
                session_rois[glom] = glomerulus_contour(proj, debug=True)
    '''

    ##########################################################################
    # TODO more idiomatic way to just let the last index be any integer?
    # in case i want to compare across trials with different framerates?
    max_frames = max([nd.shape[0] for nd in deltaF])

    trial_counter = dict()
    for p in pinsetlist:
        fs = frozenset(p)
        if fs in trial_counter:
            trial_counter[fs] += 1
        else:
            trial_counter[fs] = 1

    max_trials = max(trial_counter.values())
    condition, fly_id, block  = u.get_fly_info(d)

    # TODO get max # of blocks and use that. index them from 1.

    # TODO use date and int for order instead of fly_id to impose ordering?
    # though there is string ordering
    # TODO load information about rearing age and air / stuff stored in external
    # csv to add to dataframe?
    # TODO verify_integrity argument

    samples_per_sec = 5
    duration = params['total_recording_s']
    timestep = 1 / samples_per_sec
    num_timepoints = samples_per_sec * duration

    multi_index = pd.MultiIndex.from_product( \
            [[condition], [fly_id], stimulus_panel, \
            range(max_trials), range(num_timepoints)], \
            names=['condition', 'fly_id', 'odor', 'trial', 'timepoint'])

    # TODO handle consistently with other rois / contours
    label2ijroi = u.get_ijrois(d, maxdF[0].shape)
    #columns = list(session_rois.keys()) 
    columns = ['manual_' + l for l in label2ijroi.keys()]

    # + ['block']

    block_df = pd.DataFrame(index=multi_index, columns=columns)
    ##########################################################################

    '''
    for glom, contour in session_rois.items():
        traces = [extract(dF, contour) for dF in deltaF]
        trial = 0

        for mixture, trace in zip(odorsetlist, traces):
            resampled = np.interp(np.linspace(0, trace.shape[0], num=num_timepoints), \
                    np.linspace(0, trace.shape[0], num=trace.shape[0]), trace)
            # flatten will go across rows before going to the next column
            # which should be the behavior we want
            block_df[glom].loc[condition, fly_id, mixture, trial] = resampled

            # TODO just put this back in the index?
            #block_df['block'].loc[condition, fly_id, mixture, trial] = block

            trial = (trial + 1) % repeats

        # TODO TODO still check this, just in the appropriate place
        # making sure we have distinct data in each column
        # and have not copied it all from one place
        #assert np.sum(np.isclose(profiles[0,:], profiles[1,:])) < 2

    '''
    for label, mask in label2ijroi.items():
        traces = [extract_mask(dF, mask) for dF in deltaF]
        trial = 0

        session_rois[label] = mask

        for mixture, trace in zip(odorsetlist, traces):
            resampled = np.interp(np.linspace(0, trace.shape[0], num=num_timepoints), \
                    np.linspace(0, trace.shape[0], num=trace.shape[0]), trace)

            block_df['manual_' + label].loc[condition, fly_id, mixture, trial] = resampled

            trial = (trial + 1) % repeats

    projections[d] = session_projections
    rois[d] = session_rois
    return block_df


# TODO memorize?
def process_experiment(exp_dir, substring2condition, params, cargs=None):
    # TODO assert that no directory has two manual rois for the same thing
    # TODO have fly_id include session, or explicitly add that index
    # TODO have rois only be in one format (image, probably)
    """
    Args:
        exp_dir: directory containing directories organized the following way: 
            -a TIFF named as /path/to/exp_dir/<imaging_dir_name>/<imaging_dir_name>_ChanA.tif
            -the Experiment.xml generated from ThorSync
            -a subdirectory with ThorSync data (the HDF5 and XML metadata)
            -pickle files holding stimulus metadata
                -generated_pin2odor.p, holding a list of triples of form

                [(arduino pin (solenoid valve), tuple description of odor, 
                  letter of manifold port), ...]

                -generated_stimorder.p, holding a list of frozensets that contain the pin(s)
                 used on the trial indexed the same
                
            -files ending in .roi saved individually from ImageJ ROI Manager (polygonal ROIs work)
                -if the part of the name before .roi includes the name of a glomerulus
                 you are interested in as a substring, it will be contribute to that glomerulus
            
            -possibly cache files generated by previous runs of this analysis
            -other files should be ignored
        
        substring2condition: dict mapping substrings, that are only present in directories
            belonging to some experimental condition you want to group on, to a description
            of that condition, which will be used as the key for the 'condition' index in
            the DataFrame to be returned

        params:
            -dict holding information about stimulus parameters to group the stack appropriately
             and set the scales
            -TODO generate somehow in the future -> save in another pickle file?
    
    Returns:
        projections: a dict of directory names to dicts taking odors to maximum intensity 
            projections of the mean response to that odor in the data in that directory. 
            terminal values are 2d numpy arrays holding images summarizing trials.

        rois: nested dicts indexed same as above, but holding either OpenCV contours 
            or images at the bottom level

        df: pandas DataFrame multindexed by rearing condition, fly_id, session #, glomerulus, 
            odor, presentation # (out of block), frame #. Contains signal binned in ROI
            detected for the specific glomerulus.
    """

    if cargs is None:
        test = False
        print_summary_only = False
        recheck_data = False
        recompute = False

    else:
        # TODO pythonic way to split up dict? keep it?
        test = cargs.test
        print_summary_only = cargs.print_summary_only 
        recheck_data = cargs.recheck_data
        recompute = cargs.recompute

    good_session_dirs = u.get_sessiondirs(exp_dir)
    print(good_session_dirs)

    if test:
        print('only running on first dir because you ran with test flag')
        session_dirs = [d for d in good_session_dirs if '170308_02c_dl5' in d]

    else:
        # TODO make this compatible with test flag somehow or make test flag imply force recheck
        # TODO make Memorize/percache work with list and allow for clearing cache
        #if ... recheck_data:

        # filter out sessions with something unexpected about their data
        # (missing frames, too many frames, wrong ITI, wrong odor pulse length, etc)
        session_dirs = checks.consistent_dirs(good_session_dirs, params)

    # TODO
    #reasons = dict()
    #session_dirs = [d for d in good_session_dirs if check_consistency(d, params, fail_store=reasons)]
    # condition -> list of session directory dicts
    projections = dict()
    rois = dict()

    df = pd.DataFrame()
    # would handle df the same way, but none of df operations seem to let 
    # me modify it in place very easily
    data_stores = (projections, rois)

    for d in sorted(session_dirs):
        if print_summary_only:
            u.print_odor_order(d, params)
        else:
            session_df = process_session(d, data_stores, params, recompute=recompute)
            df = df.append(session_df)

    print('\nInitially considered:')
    for s in good_session_dirs:
        print(s)

    print('\nMet data standards:')
    for s in session_dirs:
        print(s)
    print('')

    # TODO print reasons remainder have failed
    return projections, rois, df
