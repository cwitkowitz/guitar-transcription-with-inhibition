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
jams_dir = os.path.join(base_dir, 'jams_test')

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

# Zip the paths and files together
zipped_files = list(zip(tracked_paths, tracked_files))

# Shuffle the files
random.shuffle(zipped_files)

# Unpack the shuffled paths/files
tracked_paths, tracked_files = zip(*zipped_files)

# Determine the index where the files should be divided to get a 90/10 split
split_idx = int(round(0.90 * len(tracked_files)))

# Split the list of files according the the index
train_files, val_files = tracked_files[:split_idx], tracked_files[split_idx:]

# Keep track of the jams files for each split
train_files_, val_files_ = list(), list()

# Loop through the GuitarPro files in the current directory
for i, f in enumerate(tracked_files):
    # Obtain a label for the song
    song_name = os.path.splitext(f)[0]

    print(f'Processing track \'{song_name}\'...')

    # Construct a path to the GuitarPro file
    gpro_path = os.path.join(tracked_paths[i], f)

    # Extract the GuitarPro data from the file
    gpro_data = guitarpro.parse(gpro_path)

    # Loop through the instrument tracks in the GuitarPro data
    for t, gpro_track in enumerate(gpro_data.tracks):
        # Make sure the GuitarPro file can be processed for symbolic tablature
        if validate_gpro_track(gpro_track):
            # Add the track number to the file name
            track_name = f'{song_name} - {t + 1}'
            # Construct a path to the JAMS file to be created
            jams_path = os.path.join(jams_dir, f'{track_name}.{tools.JAMS_EXT}')

            # TODO - cleaned up following two functions
            # Extract notes from the track, given the listed tempo
            notes_per_string = get_all_notes_from_track(gpro_track, gpro_data.tempo)
            # Write the JAMS files
            # TODO - if the track is not completely silent
            convert_notes_per_string_to_jams(notes_per_string, jams_path)

            # Add the track name to the proper list
            if f in train_files:
                train_files_ += [track_name]
            else:
                val_files_ += [track_name]

# Write the track names to respective text files
tools.write_list(train_files_, os.path.join(base_dir, 'train_split.txt'))
tools.write_list(val_files_, os.path.join(base_dir, 'val_split.txt'))
