# Author: Jonathan Driedger (Chordify)
# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

from guitarpro_jams_utils import validate_gpro_track, extract_stacked_notes_gpro_track

# Regular imports
import numpy as np
import guitarpro
import os

# TODO - add to a constants.py?
VALID_GP_EXTS = ['.gp3', '.gp4', '.gp5']
INVALID_EXTS = ['.pygp', '.gp2tokens2gp']
COPY_TAG = ' copy'


def get_valid_files(base_dir, ignore_duplicates=True):
    """
    Walk through a base directory and keep track of all relevant GuitarPro files.

    Parameters
    ----------
    base_dir : string
      Path to the base directory to recursively search
    ignore_duplicates : bool
      Whether to remove exact duplicates and inferred duplicates

    Returns
    ----------
    tracked_files : list of str
      List of file names found
    tracked_paths : list of str
      List of paths corresponding to tracked files
    """

    # Keep track of valid GuitarPro files
    tracked_paths, tracked_files = list(), list()

    # Traverse through all paths within the base directory
    for dir_path, dirs, files in os.walk(base_dir):
        # Ignore directories with no files (only directories)
        if not len(files):
            continue

        # Obtain a list of valid GuitarPro files within the current directory
        valid_files = sorted([f for f in files
                              if os.path.splitext(f)[-1] in VALID_GP_EXTS
                              and INVALID_EXTS[0] not in f
                              and INVALID_EXTS[1] not in f
                              # Remove (exact) duplicates
                              and not (f in tracked_files and ignore_duplicates)])

        # Remove (inferred) duplicates within the directory
        if ignore_duplicates:
            # Obtain a list of copied files
            copied_files = [f for f in valid_files if COPY_TAG in f]

            # Loop through copies in the directory
            for f in copied_files:
                # Determine the name of the original file
                f_name, ext = os.path.splitext(f)[0][:-len(COPY_TAG)], os.path.splitext(f)[-1]
                # Construct paths to the copy and original
                copy_path = os.path.join(dir_path, f)
                orig_path = os.path.join(dir_path, f_name + ext)

                # Remove copies if they have the same file size
                # TODO - should we make this specification?
                # if os.path.getsize(copy_path) == os.path.getsize(orig_path):
                valid_files.remove(f)

            # Create a copy of the valid files to iterate through
            valid_files_copy = valid_files.copy()

            # Loop through the current valid files list
            for i in range(0, len(valid_files) - 1):
                # Obtain the current and next valid file
                curr_file, next_file = valid_files_copy[i], valid_files_copy[i + 1]
                # Check if the two files share the same name
                if os.path.splitext(curr_file)[0] == os.path.splitext(next_file)[0]:
                    # Remove the current file (should be earlier version)
                    valid_files.remove(curr_file)

        # Add valid files to tracked list
        tracked_files += valid_files

        # Update the tracked paths
        tracked_paths += [dir_path] * len(valid_files)

    return tracked_files, tracked_paths


def guitarpro_to_jams(gpro_path, jams_dir):
    """
    Convert a GuitarPro file to a JAMS file specifying notes for each track.
    TODO - add others stuff (beats, key, tempo, etc.) to the JAMS files

    Parameters
    ----------
    gpro_path : string
      Path to a preexisting GuitarPro file to convert
    jams_dir : bool
      Directory under which to place the JAMS files
    """

    # Make sure the JAMS directory exists
    os.makedirs(jams_dir, exist_ok=True)

    # Extract the GuitarPro data from the file
    gpro_data = guitarpro.parse(gpro_path)

    # Loop through the instrument tracks in the GuitarPro data
    for t, gpro_track in enumerate(gpro_data.tracks):
        # Make sure the GuitarPro file can be processed for symbolic tablature
        if validate_gpro_track(gpro_track):
            # Add the track number to the file name
            track_name = f'{gpro_file} - {t + 1}'
            # Construct a path to the JAMS file to be created
            jams_path = os.path.join(jams_dir, f'{track_name}.{tools.JAMS_EXT}')

            # Extract notes from the track, given the listed tempo
            stacked_notes = extract_stacked_notes_gpro_track(gpro_track, gpro_data.tempo)

            if np.sum([len(stacked_notes[key][0]) for key in stacked_notes.keys()]):
                # Write the JAMS files if it is not completely silent
                tools.write_stacked_notes_jams(stacked_notes, jams_path)
                # TODO - dealing with negative duration?


if __name__ == '__main__':
    # Construct a path to the base directory
    #base_dir = 'path/to/DadaGP'
    base_dir = os.path.join(tools.HOME, 'Desktop', 'Datasets', 'DadaGP')

    # Search the specified path for GuitarPro files
    tracked_files, tracked_paths = get_valid_files(base_dir)

    # Construct a path to the JAMS directory
    jams_dir = os.path.join(base_dir, 'jams')

    # Loop through the tracked GuitarPro files
    for k, gpro_file in enumerate(tracked_files):
        # TODO - remove
        # Testing files
        #if not ('Suck My Kiss' in gpro_file):
        #if not ('Pink Floyd - If' in gpro_file):
        #if not ('Nothing else matters (7)' in gpro_file):
        if not ('Prewar - Song Of War.gp3' in gpro_file):
            continue
        # Error files
        #if not ('Pillows (The) - Funny Bunny.gp4' in gpro_file):
        #if not ('Perkins, Carl - Matchbox.gp4' in gpro_file):
        #    continue

        print(f'Processing track \'{gpro_file}\'...')

        # Construct a path to the GuitarPro file
        gpro_path = os.path.join(tracked_paths[k], gpro_file)

        # Perform the conversion
        guitarpro_to_jams(gpro_path, jams_dir)
