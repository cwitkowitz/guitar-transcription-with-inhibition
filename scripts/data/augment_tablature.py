# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.transcribe import filter_notes_by_duration

import amt_tools.tools as tools

# Regular imports
from copy import deepcopy

import numpy as np


def random_capo_move(stacked_notes, profile, rng=None):
    """
    Randomly shift the relative fret positions by using an invisible capo.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    profile : GuitarProfile (instrument.py)
      Instrument profile detailing experimental setup
    rng : NumPy RandomState
      Random number generator to use for augmentation

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    if rng is None:
        # Default the random state
        rng = np.random

    # Get the MIDI pitches of the open strings
    midi_tuning = profile.get_midi_tuning()

    # Determine the minimum and maximum pitch played on each string
    min_pitches, max_pitches = tools.find_pitch_bounds_stacked_notes(stacked_notes)

    # Default the number of unused frets
    unused_frets_left = profile.get_num_frets() * np.ones(midi_tuning.shape)
    unused_frets_right = profile.get_num_frets() * np.ones(midi_tuning.shape)

    # Determine how many frets are unused along both directions of fretboard
    unused_frets_left[min_pitches != 0] = (min_pitches - midi_tuning)[min_pitches != 0]
    unused_frets_right[max_pitches != 0] = (midi_tuning - max_pitches)[max_pitches != 0] + profile.get_num_frets()

    # Determine the maximum capo shift in both directions
    max_capo_left = max(0, np.min(unused_frets_left))
    max_capo_right = max(0, np.min(unused_frets_right))

    if max_capo_left != 0 or max_capo_right != 0:
        # Randomly sample a capo shift
        fret_shift = rng.randint(-max_capo_left, max_capo_right + 1)

        # Add a semitone offset to all of the pitches in the stacked notes to model the capo shift
        stacked_notes = tools.apply_func_stacked_representation(stacked_notes, tools.offset_notes, semitones=fret_shift)

    return stacked_notes


def random_notes_drop(stacked_notes, drop_rate, rng=None):
    """
    Randomly drop notes.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    drop_rate : float
      Approximate percentage of notes to drop
    rng : NumPy RandomState
      Random number generator to use for augmentation

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    if rng is None:
        # Default the random state
        rng = np.random

    # Make a copy of the stacked notes for conversion
    stacked_notes = deepcopy(stacked_notes)

    # Loop through the stack of notes
    for i, slc in enumerate(stacked_notes.keys()):
        # Get the notes from the slice as batched notes
        batched_notes = tools.notes_to_batched_notes(*stacked_notes[slc])

        # Randomly sample the notes indices to drop
        keep_indices = rng.rand(batched_notes.shape[-2]) > drop_rate

        # Remove the dropped notes, convert back to regular notes, and add back to the stack
        stacked_notes[slc] = tools.batched_notes_to_notes(batched_notes[keep_indices])

    return stacked_notes


def random_onset_offset_shift(stacked_notes, std_dev, rng=None):
    """
    Randomly shift the onset and offset of notes across time.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    std_dev : float
      Standard deviation of shifts
    rng : NumPy RandomState
      Random number generator to use for augmentation

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    if rng is None:
        # Default the random state
        rng = np.random

    # Make a copy of the stacked notes for conversion
    stacked_notes = deepcopy(stacked_notes)

    # Loop through the stack of notes
    for i, slc in enumerate(stacked_notes.keys()):
        # Extract the note groups for the slice
        pitches, intervals = stacked_notes[slc]

        # Sample random noise to add to the note intervals
        noise = rng.normal(scale=std_dev, size=intervals.shape)

        # Add the noise to the intervals
        intervals += noise

        # Make sure the notes are sorted by onset
        pitches, intervals = tools.sort_notes(pitches, intervals)

        # Initialize an array to hold differentials
        diffs = None

        # Keep adjusting until there is no overlap
        while diffs is None or np.sum(diffs):
            if diffs is not None:
                # Reshape the differentials and add their magnitude back to the intervals to fix overlap
                intervals -= np.reshape(diffs, intervals.shape)

            # Compute the differentials across the sorted intervals
            diffs = intervals.flatten()[1:] - intervals.flatten()[:-1]

            # Append zero for the first index
            diffs = diffs if pitches.size == 0 else np.append([0], diffs)

            # Positive differentials mean there are no problems
            diffs[diffs >= 0] = 0

        # TODO - clip notes at 0 and original duration?

        # Make sure there are no notes with zero duration after the previous operations
        pitches, intervals = filter_notes_by_duration(pitches, intervals)

        # Add the augmented notes back to the stack
        stacked_notes[slc] = pitches, intervals

    return stacked_notes
