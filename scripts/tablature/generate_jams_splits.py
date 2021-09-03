# Author: Jonathan Driedger (Chordify)
# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

from guitarpro_jams_utils import validate_gpro_track, get_all_notes_from_track, convert_notes_per_string_to_jams

# Regular imports
import guitarpro
import random
import os

VALID_GP_EXTS = ['.gp3', '.gp4', '.gp5']
INVALID_EXTS = ['.pygp', '.gp2tokens2gp']

# Set the seed for random number generation
random.seed(0)

# Construct a path to the base directory
base_dir = os.path.join(tools.HOME, 'Desktop', 'Datasets', 'DadaGP')

# Construct a path to the JAMS directory
jams_dir = os.path.join(base_dir, 'jams')

# Make sure the JAMS directory exists
os.makedirs(jams_dir, exist_ok=True)

# Keep track of valid GuitarPro files
tracked_paths, tracked_files = list(), list()

# Traverse through all paths within the base directory
for dir_path, dirs, files in os.walk(base_dir):
    # Ignore directories with no files (only directories)
    if not len(files):
        continue

    # Obtain a list of valid GuitarPro files with the current directory
    valid_files = [f for f in files
                   if os.path.splitext(f)[-1] in VALID_GP_EXTS
                   and INVALID_EXTS[0] not in f
                   and INVALID_EXTS[1] not in f
                   and f not in tracked_files]

    # Add valid files to tracked list
    tracked_files += valid_files

    # Update the tracked paths
    tracked_paths += [dir_path] * len(valid_files)

# Loop through the tracked GuitarPro files
# TODO - could probably combine the for-loops now, even possibly have a jams/ directory per sub-split
for i, gpro_file in enumerate(tracked_files):
    print(f'Processing track \'{gpro_file}\'...')

    # Construct a path to the GuitarPro file
    gpro_path = os.path.join(tracked_paths[i], gpro_file)

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

            # TODO - cleaned up following two functions
            try:
                # Extract notes from the track, given the listed tempo
                notes_per_string = get_all_notes_from_track(gpro_track, gpro_data.tempo)
                # Write the JAMS files
                # TODO - only if the track is not completely silent
                # TODO - dealing with negative duration?
                convert_notes_per_string_to_jams(notes_per_string, jams_path)
            except:
                continue
