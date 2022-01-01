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

remove_duplicates = True

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

    # Obtain a list of valid GuitarPro files within the current directory
    valid_files = sorted([f for f in files
                          if os.path.splitext(f)[-1] in VALID_GP_EXTS
                          and INVALID_EXTS[0] not in f
                          and INVALID_EXTS[1] not in f
                          and not (f in tracked_files and remove_duplicates)])

    # Remove duplicates within the directory
    if remove_duplicates:
        # Obtain a list of copied files
        copied_files = [f for f in valid_files if ' copy' in f]

        # Loop through copies in the directory
        for f in copied_files:
            # Determine the name of the original file
            f_name, ext = os.path.splitext(f)[0][:-5], os.path.splitext(f)[-1]
            # Construct paths to the copy and original
            copy_path = os.path.join(dir_path, f)
            orig_path = os.path.join(dir_path, f_name + ext)

            # Remove copies if they have the same file size
            # TODO - should we make this specification?
            #if os.path.getsize(copy_path) == os.path.getsize(orig_path):
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

# Loop through the tracked GuitarPro files
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
