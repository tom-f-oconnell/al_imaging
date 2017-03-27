#!/usr/bin/env python3

import os
import thunder as td

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

def new_tif_name(directory):
    if not os.path.isdir(directory):
        raise IOError('tried to generate a new tif name from something ' + \
                'that was not a directory. pass directory holding multiple ' + \
                'individual tifs that you want to convert to one single file.'

    frags = os.path.split(directory)
    return os.path.join(*frags, frags[-1] + '.tif')


# TODO when will the first number not be a zero?
# using that to exclude the ..._Preview... tif
def convert_to_bare_tiff(ome_tiff_directory):

    # files you want to include in multipage tiff will need to
    globs = ['ChanA*.tif', 'ChanB*.tif']

    # save as ../ome_tiff_directory/ome_tiff_directory.tif
    
    # TODO robust metrics of success? (before deleting)
    # test before deleting

def delete_multifile_tifs(ome_tiff_directory):


def resave_metadata(first_tiff_name):
    parent_dir = os.path.join(os.path.split(first_tiff_name)[:-1])
    metadata_fname = 'extracted_metadata.txt'
    meta_fullname = os.path.join(parent_dir, metadata_fname)
    with Image() as i:
        with open(meta_fullname, 'w') as f:
            
def test_im_concat():
    parent = get_script_path()
    fname = 'example_multifile_tiff'
    test_dir = os.path.join(parent, fname)
    assert os.path.isdir(test_dir)

    from_multi = td.images.fromtif(test_dir)
    convert_to_bare_tiff(test_dir)
    from_converted = td.images.fromtif()


if __name__ == "__main__":
    thorimage_directory = '/media/threeA/Tom/flies/test/mp'

