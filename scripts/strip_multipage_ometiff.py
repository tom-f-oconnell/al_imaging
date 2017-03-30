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

    files = os.listdir(directory)
    def in_some_filename(substring):
        for f in files:
            if substring in f:
                return True
        return False

    for f in files:
        for p in channel_prefixes:
            if p + '_' in f and '.tif' in f and not (p + '_0') in f and not '_Preview.tif' in f:
                print('problematic file:', f)
                raise AssertionError('need to exclude preview tif some other way')

    return [p for p in channel_prefixes if in_some_filename(p)]


def get_channel_tiffs(d, c):
    return [join(d, f) for f in os.listdir(d) if c + '_' in f and '.tif' in f and not 'Preview' in f]


def strip_tiffs_inplace(thorimage_directory, channel_prefixes):
    
    print('stripping metadata from all files in', thorimage_directory, \
            'from channels', channel_prefixes, '...')

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
    

def concatenate_tiffs(thorimage_directory, channel_prefixes):
    # save as ../thorimage_directory/thorimage_directory_<channel>.tif
    for p in channel_prefixes:
        # should now be able to load the sequence with tifffile,
        # not having to also load GB of header data
        # script should not make it to this point (should fail in getting channels)
        # if this glob, starting with <channel>_0 is missing some tifs
        sequence = tifffile.imread(join(thorimage_directory, p + '_0' + '*.tif'))

        '''
        # move Z dimension to the 3rd
        sequence = np.swapaxes(sequence, 2, 0)
        # restore order of X and Y
        sequence = np.swapaxes(sequence, 1, 0)

        # add unit dimensions for C and Z channels
        assert len(sequence.shape) == 3, 'more dimensions than expected'
        sequence = np.expand_dims(sequence, 3)
        sequence = np.expand_dims(sequence, 4)
        sequence = np.swapaxes(sequence, 2, 4)

        print(sequence.shape)
        '''

        #tifffile.imsave(new_tiff_name(thorimage_directory, p), sequence, metadata={'axes': 'XYCZT'})

        # using this because Thunder doesn't seem to want to write single file tiffs
        # although they should have generally have less overhead than multifile
        tifffile.imsave(new_tiff_name(thorimage_directory, p), sequence)


'''
def get_td_summary(load_arg):
    # TODO be more careful with volumes over time
    imgs = td.images.fromtif(load_arg).toarray().squeeze()
    return imgs.shape, imgs[0,:,:], imgs[1,:,:], imgs[-2,:,:], imgs[-1,:,:]
'''


def first_diff_frame(imgs1, imgs2):
    min_len = min(imgs1.shape[0], imgs2.shape[0])
    for i in range(min_len):
        if not np.all(np.equal(imgs1[i,:,:], imgs2[i,:,:])):
            return i

    if imgs1.shape[0] != imgs2.shape[0]:
        return min_len

    # they don't differ
    return -1


def same_set_of_frames(imgs1, imgs2):
    s1 = set()
    for i in range(imgs1.shape[0]):
        s1.add(np.mean(imgs1[i,:,:]))

    s2 = set()
    for i in range(imgs2.shape[0]):
        s2.add(np.mean(imgs2[i,:,:]))

    print(imgs1.shape)
    print(imgs2.shape)
    print(len(s1))
    print(len(s2))
    print(s1 == s2)


def check_conversion(thorimage_directory, channel_prefixes):
    print('checking conversion was successful... ', end='')
    for p in channel_prefixes:
        # should be able to load original multifile tiffs because bulky headers were stripped
        '''
        os, ofirst, osec, oslast, olast = \
                get_td_summary(join(thorimage_directory, p + '_0' + '*.tif'))
        ns, nfirst, nsec, nslast, nlast = get_td_summary(new_tiff_name(thorimage_directory, p))

        first_equal = np.all(np.equal(ofirst, nfirst))
        sec_equal = np.all(np.equal(osec, nsec))
        slast_equal = np.all(np.equal(oslast, nslast))
        last_equal = np.all(np.equal(olast, nlast))

        toshow = [ofirst, nfirst, oslast, nslast, olast, nlast]
        for i in toshow:
            plt.figure()
            plt.imshow(i)
        plt.show()
        '''
        # TODO so it seems like the concatenating process is causing the difference?
        old = td.images.fromtif(join(thorimage_directory, p + '_0' + '*.tif'))
        print(old.shape)
        old = old.toarray().squeeze()

        new = td.images.fromtif(new_tiff_name(thorimage_directory, p))
        print(new.shape)
        new = new.toarray().squeeze()

        for i in range(min(new.shape[0], old.shape[0])):
            print(i, np.sum(np.abs(new[i,:,:] - old[i,:,:])))

        first_diff = first_diff_frame(old, new)
        print(same_set_of_frames(old, new))
        assert old.shape == new.shape and first_diff == -1, \
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
        #    raise AssertionError('extracted metadata already exists! stopping to not overwrite it.')


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
    if resave_metadata(d, cps):
        strip_tiffs_inplace(d, cps)

    concatenate_tiffs(d, cps)
    check_conversion(d, cps)
    #delete_multifile_tiffs(d, cps)

            
def test_roundtrip():
    # TODO
    pass


if __name__ == "__main__":
    #d = '/media/threeA/Tom/flies/test/mp'
    #d = '/media/threeA/Tom/flies/170319_02c_dm6'
    d = '/media/threeA/Tom/flies/01_003'
    convert(d)
    
