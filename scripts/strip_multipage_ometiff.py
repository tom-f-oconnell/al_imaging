#!/usr/bin/env python3

from __future__ import print_function

import os
from os.path import join, split, isfile, isdir
import numpy as np
from wand.image import Image
import tifffile
import thunder as td

import matplotlib.pyplot as plt

# TODO logging

def get_script_path():
    return dirname(realpath(__file__))


def get_channels(directory):
    channel_prefixes = ['ChanA', 'ChanB', 'ChanC']

    # files output in ThorImage naming convention have 4 groups of (always? zstack?)
    # numbers separated by underscores
    # TODO use this in get_channel_tiffs? refactor to make these two functions share more code,
    # for consistencies sake?
    def has_numbers(fname):
        def has_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        return len([int(s) for s in fname.replace('.tif','').split('_') if has_int(s)]) == 4

    files = os.listdir(directory)
    def valid_multifile(substring):
        num_tiffs = 0
        for f in files:
            if '.tif' in f and substring in f and not 'Preview' in f and has_numbers(f):
                num_tiffs += 1

            if num_tiffs > 1:
                return True

        return False

    return [p for p in channel_prefixes if valid_multifile(p)]


def get_channel_tiffs(d, c):
    return [join(d, f) for f in os.listdir(d) if c + '_' in f and '.tif' in f and not 'Preview' in f]


def strip_tiffs_inplace(thorimage_directory, channel_prefixes):
    
    print('stripping metadata from all files in', thorimage_directory, \
            'from channels', channel_prefixes, '...')

    # TODO make sure this is consistent with list of files from all channels
    files = [join(thorimage_directory, f) for f in os.listdir(thorimage_directory) \
            if any([p + '_' in f for p in channel_prefixes]) and '.tif' in f]

    for f in files:
        with Image(filename=f) as i:
            i.strip()
            img = i
            img.save(filename=f)


def new_tiff_name(d, c):
    frags = split(d)
    return join(*frags, frags[-1] + '_' + c + '.tif')


def load_frame(framename):
    frame = tifffile.imread(framename)
    return frame


def last_number(filename):
    return int(split(filename)[-1].split('_')[-1][:-4])


# TODO test. test consistency. and with other loading methods. + roundtrip
def load_multifile_tiff(d, channel):
    files = get_channel_tiffs(d, channel)
    if len(files) == 0:
        return None

    # it is really important to sort based on the last integer
    # because lexicographic sorting (default for strings) will misorder frames in trials
    # with >= 10k frames, with the naming convention ThorImage uses
    sorted_files = sorted(files, key=last_number)

    # use dims/dtype?
    arr = td.images.fromlist(sorted_files, accessor=load_frame).toarray()
    return np.squeeze(arr)


def load_singlefile_tiff(fname):
    arr = td.images.fromtif(fname).toarray()
    return np.squeeze(arr)


def save_array_to_singletiff(arr, fname):
    #tifffile.imsave(fname, arr, metadata={'axes': 'XYCZT'})

    # using this because Thunder doesn't seem to want to write single file tiffs
    # although they should have generally have less overhead than multifile
    tifffile.imsave(fname, arr)
    

def concatenate_tiffs(d, channel_prefixes):
    print('concatenating multifile tiffs in', d)

    # save as ../thorimage_directory/thorimage_directory_<channel>.tif
    for p in channel_prefixes:
        print(p)
        # should now be able to load the sequence,
        # not having to also load GB of header data
        sequence = load_multifile_tiff(d, p)
        if sequence is None:
            continue
        save_array_to_singletiff(sequence, new_tiff_name(d, p))


def first_diff_frame(imgs1, imgs2):
    min_len = min(imgs1.shape[0], imgs2.shape[0])
    for i in range(min_len):
        if not np.all(np.equal(imgs1[i,:,:], imgs2[i,:,:])):
            return i

    if imgs1.shape[0] != imgs2.shape[0]:
        return min_len

    # they don't differ
    return -1


def stacks_equal(imgs1, imgs2):
    # assumes they are either of type td.images.Image or np.ndarray
    if type(imgs1) is not np.ndarray:
        imgs1 = imgs1.toarray()
    if type(imgs2) is not np.ndarray:
        imgs2 = imgs2.toarray()

    imgs1 = imgs1.squeeze()
    imgs2 = imgs2.squeeze()

    return imgs1.shape == imgs2.shape and np.all(np.equal(imgs1, imgs2))


def check_conversion(d, channel_prefixes):
    print('checking conversion was successful... ', end='')
    for p in channel_prefixes:
        # should be able to load original multifile tiffs because bulky headers were stripped
        old = load_multifile_tiff(d, p)
        new = load_singlefile_tiff(new_tiff_name(d, p))

        first_diff = first_diff_frame(old, new)
        assert stacks_equal(old, new), \
                '\nnot reading same data after conversion.\n old shape: ' + str(old.shape) + \
                ' new shape: ' + str(new.shape) + '\nfirst different frame: ' + str(first_diff)+ '\n'

    print('OK')
    return True


def delete_multifile_tiffs(d, cps):
    for p in cps:
        print('removing sequence of single frame tifs for channel', p, '...')
        # TODO test it is f and not join(d, f) or something
        for f in get_channel_tiffs(d, p):
            os.remove(f)
        

def resave_metadata(thorimage_directory, channel_prefixes):
    print('extracting metadata from the first tiff of each channel...')

    metadata_suffix = '_extracted_metadata.txt'
    for p in channel_prefixes:
        print(p)
        sorted_files = sorted(get_channel_tiffs(thorimage_directory, p))
        first_tiff = join(thorimage_directory, sorted_files[0])

        assert first_tiff[-4:] == '.tif', 'extension ' + first_tiff[-4:] + ' not .tif'
        assert first_tiff[-8:-4] == '0001', 'first file in list sorted by name not 0001: ' + \
            sorted_files[:4]

        meta_fullname = join(thorimage_directory, p + metadata_suffix)

        if isfile(meta_fullname):
            print('not overwriting extracted metadata')
            return False

        with Image(filename=first_tiff) as i:
            print('writing metadata to', meta_fullname, '...')
            with open(meta_fullname, 'w') as f:
                for e in i.metadata:
                    print(e, file=f)
                    print(i.metadata[e], file=f)

        assert isfile(meta_fullname), 'save not successful'

    return True


def convert(d):
    cps = get_channels(d)

    # only tries stripping if extracted metadata isn't already there
    # mostly to save time in debugging
    #if resave_metadata(d, cps):
    #    strip_tiffs_inplace(d, cps)

    concatenate_tiffs(d, cps)
    check_conversion(d, cps)
    #delete_multifile_tiffs(d, cps)


def test_last_number():
    fname = 'ChanA_0001_0001_0001_9999.tif'
    assert last_number(fname) == 9999


def test_tifffile_load_frame(dims):
    fname = 'test_tiffile_load_frame.DELETEME.tif'

    for d in dims:
        a = np.random.rand(*d)
        save_array_to_singletiff(a, fname)
        b = load_frame(fname)
        os.remove(fname)
        assert stacks_equal(a, b)


def test_loadsave(dims):
    fname = 'test_loadsave.DELETEME.tif'
    
    for d in dims:
        a = np.random.rand(*d)
        save_array_to_singletiff(a, fname)
        b = load_singlefile_tiff(fname)
        os.remove(fname)
        assert stacks_equal(a, b)


def test_concat():
    # TODO
    raise NotImplementedError


# TODO it looks like i deleted my work checking consistency of thunder and tifffile
# REDO THAT w/ various sortings of inpu, and list vs non-list
            
def run_tests():
    print('running some tests')

    test_last_number()

    dims = [(1, 400, 500), (400, 500, 1), (1, 1, 400, 500), \
            (1, 400, 500, 1, 1), (400, 500), (1, 128, 128)]
    test_tifffile_load_frame(dims)
    test_loadsave(dims)

    print('tests passed')


if __name__ == "__main__":
    run_tests()

    parent_directory = '/media/threeA/Tom/flies'
    dirs = [join(parent_directory, d) for d in os.listdir(parent_directory) \
            if isdir(join(parent_directory, d))]

    for d in dirs:
        print(d)
        convert(d)
