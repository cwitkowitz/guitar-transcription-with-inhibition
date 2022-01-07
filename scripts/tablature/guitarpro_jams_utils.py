# Author: Jonathan Driedger (Chordify)
# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

# Regular imports
import numpy as np
import librosa

TICKS_PER_QUARTER_NOTE = 960
NOTE_TYPE_ENUM_REST    = 'rest'
NOTE_TYPE_ENUM_NORMAL  = 'normal'
NOTE_TYPE_ENUM_TIE     = 'tie'
NOTE_TYPE_ENUM_DEAD    = 'dead'
DURATION_SCALE         = 1.0


def ticks_to_seconds(ticks, tempo):
    """
    Convert an amount of ticks to a concrete time.

    Parameters
    ----------
    ticks : int or float
      Amount of ticks
    tempo : int or float
      Number of beats per minute

    Returns
    ----------
    time : float
      Time in seconds corresponding to number of ticks
    """

    # Number of seconds per beat times number of quarter beats
    time = (60 / tempo) * ticks / TICKS_PER_QUARTER_NOTE

    return time


class Note(object):
    """
    Simple class representing a guitar note for use during GuitarPro file processing.
    """

    def __init__(self, fret, onset, duration, string=None):
        """
        Initialize a guitar note.

        Parameters
        ----------
        fret : int
          Fret the note was played on
        onset : float
          Time of the beginning of the note in seconds
        duration : float
          Amount of time after the onset where the note is still active
        string : int (Optional)
          Numerical indicator for the string the note was played on
        """

        self.fret = fret
        self.onset = onset
        self.duration = duration
        self.string = string

    def extend_note(self, duration):
        """
        Extend the note by a specified amount of time.

        Parameters
        ----------
        duration : float
          Amount of time to extend the note
        """

        self.duration = duration


class NoteTracker(object):
    """
    Simple class to keep track of state while tracking notes in a GuitarPro file.
    """

    def __init__(self, default_tempo, tuning=None):
        """
        Initialize the state of the tracker.

        Parameters
        ----------
        default_tempo : int or float
          Underlying tempo of the track
        tuning : list or ndarray
            MIDI pitch of each open-string
        """

        # Keep track of both the underlying and current tempo
        self.default_tempo = default_tempo
        self.current_tempo = default_tempo

        if tuning is None:
            # Default the guitar tuning
            tuning = librosa.note_to_midi(tools.DEFAULT_GUITAR_TUNING)

        # Keep track of the tuning and string names
        self.string_keys = librosa.midi_to_note(tuning)

        # Dictionary to hold all notes
        self.stacked_gpro_notes = dict()
        # Loop though the tuning from lowest to highest note
        for pitch in sorted(tuning):
            # Determine the corresponding key for the string
            key = librosa.midi_to_note(pitch)
            # Add an empty list as an entry for the string
            self.stacked_gpro_notes[key] = list()

    def set_current_tempo(self, tempo=None):
        """
        Update the currently tracked tempo.

        Parameters
        ----------
        tempo : int or float (Optional)
          New tempo
        """

        if tempo is None:
            # Reset the current tempo to the default
            self.current_tempo = self.default_tempo
        else:
            # Update the current tempo
            self.current_tempo = tempo

    def get_current_tempo(self):
        """
        Obtain the currently tracked tempo.

        Returns
        ----------
        tempo : int or float (Optional)
          Current tracked tempo
        """

        tempo = self.current_tempo

        return tempo

    def track_note(self, gpro_note, onset, duration):
        """
        Update the currently tracked tempo.

        Parameters
        ----------
        gpro_note : PyGuitarPro Note object
          GuitarPro note information
        onset : float
          Time the note begins in seconds
        duration : float
          Amount of time the note is active
        """

        # Extraction all relevant note information
        string_idx, fret, type = gpro_note.string - 1, gpro_note.value, gpro_note.type.name

        # TODO - determine how to deal with letRing, staccato, and other NoteEffects

        # TODO - remove this if it never happens
        if gpro_note.durationPercent != 1.0:
            print('Duration Percentage of Note != 1.0!!!!!!!!')

        # Scale the duration by the duration percentage
        duration *= gpro_note.durationPercent

        # Create a note object to keep track of the GuitarPro note
        note = Note(fret, onset, duration, string_idx)

        # Get the key corresponding to the string index
        key = self.string_keys[string_idx]

        if type == NOTE_TYPE_ENUM_NORMAL:
            # Add the new note to the dictionary under the respective string
            self.stacked_gpro_notes[key].append(note)
        elif type == NOTE_TYPE_ENUM_TIE:
            # Obtain the last note that occurred on the string
            last_gpro_note = self.stacked_gpro_notes[key][-1] \
                             if len(self.stacked_gpro_notes[key]) else None
            # Determine if the last note should be extended
            if last_gpro_note is not None and \
               note.fret == last_gpro_note.fret:
                # Determine how much to extend the note
                new_duration = onset - last_gpro_note.onset + duration
                # Extend the previous note by the current beat's duration
                last_gpro_note.extend_note(new_duration)
        else:
            pass

    def get_stacked_notes(self):
        """
        Obtain the tracked GuitarPro notes as stacked notes.

        Returns
        ----------
        stacked_notes : dict
          Dictionary containing (slice -> (pitches, intervals)) pairs
        """

        # Initialize a new dictionary to hold the notes
        stacked_notes = dict()

        # Loop through all strings
        for key in self.stacked_gpro_notes.keys():
            # Initialize empty arrays to hold note contents
            pitches, intervals = np.empty(0), np.empty((0, 2))

            # Loop through all the GuitarPro notes for the string
            for note in self.stacked_gpro_notes[key]:
                # Add the absolute pitch of the note
                pitches = np.append(pitches, librosa.note_to_midi(key) + note.fret)
                # Scale the duration to avoid frame overlap between adjacent notes
                duration = DURATION_SCALE * note.duration
                # Add the onset and offset of the note
                intervals = np.append(intervals, [[note.onset, note.onset + duration]], axis=0)

            # Populate the dictionary with the notes for the string
            stacked_notes.update(tools.notes_to_stacked_notes(pitches, intervals, key))

        return stacked_notes


def validate_gpro_track(gpro_track, tuning=None):
    """
    Helper function to determine which GuitarPro tracks are valid for our purposes.

    Parameters
    ----------
    gpro_track : Track object (PyGuitarPro)
      GuitarPro track to validate
    tuning : list or ndarray
        MIDI pitch of each open-string

    Returns
    ----------
    valid : bool
      Whether the GuitarPro track is considered valid
    """

    if tuning is None:
        # Default the guitar tuning
        tuning = librosa.note_to_midi(tools.DEFAULT_GUITAR_TUNING)

    # Determine if this is a percussive track
    percussive = gpro_track.isPercussionTrack

    # Determine if this is a valid guitar track
    guitar = (24 <= gpro_track.channel.instrument <= 31)

    # Determine if the track has the expected number of strings in the expected tuning
    expected_strings = set(tuning) == set([s.value for s in gpro_track.strings])

    # Determine if the track is valid
    valid = not percussive and guitar and expected_strings

    return valid


def extract_stacked_notes_gpro_track(gpro_track, default_tempo):
    """
    Extract MIDI notes spread across strings within a GuitarPro track into a dictionary.

    Parameters
    ----------
    gpro_track : Track object (PyGuitarPro)
      GuitarPro track data
    default_tempo : int
      Track tempo for inferring note onset and duration

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Obtain the tuning of the strings of the track
    tuning = [string.value for string in gpro_track.strings]
    # Initialize a tracker to keep track of GuitarPro notes
    note_tracker = NoteTracker(default_tempo, tuning)

    # Keep track of the amount of time processed so far
    current_time = None

    # Determine how many measures are in the track
    total_num_measures = len(gpro_track.measures)

    # Keep track of the current measure
    current_measure = 0
    # Keep track of the last measure which opened a repeat
    repeat_measure = 0
    # Keep track of the measure after the most recently encountered repeat close
    next_jump = None

    # Initialize a counter to keep track of how many times a repeat was obeyed
    repeat_count = 0

    # Loop through the track's measures
    while current_measure < total_num_measures:
        # Process the current measure
        measure = gpro_track.measures[current_measure]

        if measure.header.repeatAlternative != 0:
            # TODO - remove this when it is fixed
            print('## Repeat Alternative ##')
            print(f'Repeat Count {repeat_count}')
            # The 'repeatAlternative' attribute seems to be encoded as binary a binary vector,
            # where the integers k in the measure header represent a 1 in the kth digit
            alt_repeat_num = np.sum([2 ** k for k in range(repeat_count)])
            # Check if it is time to jump past the repeat close
            if alt_repeat_num >= measure.header.repeatAlternative:
                repeat_count = 0
                measure.header.repeatAlternative = -1
                # Jump past the repeat
                current_measure = next_jump
                continue

        # TODO - remove when the trajectory is correct
        print(f'Current Measure: {current_measure + 1}')

        if measure.isRepeatOpen:
            # Jump back to this measure at the next repeat close
            repeat_measure = current_measure

        # Loop through voices within the measure
        for voice in measure.voices:
            # Loop through the beat divisions of the measure
            for beat in voice.beats:
                if current_time is None:
                    # Set the current time to the start of the measure
                    current_time = ticks_to_seconds(measure.start, note_tracker.get_current_tempo())

                # Check if there are any tempo changes
                if beat.effect.mixTableChange is not None:
                    if beat.effect.mixTableChange.tempo is not None:
                        # Extract the updated tempo
                        new_tempo = beat.effect.mixTableChange.tempo.value
                        # Update the tempo of the note tracker
                        note_tracker.set_current_tempo(new_tempo)

                        # TODO - remove this if it never happens
                        if beat.effect.mixTableChange.tempo.duration != 0:
                            print('Tempo Change Duration != 0!!!!!!!!')

                # Convert the note duration from ticks to seconds
                duration_seconds = ticks_to_seconds(beat.duration.time, note_tracker.get_current_tempo())

                # Loop through the notes in the beat division
                for note in beat.notes:
                    # Add the note to the tracker
                    note_tracker.track_note(note, current_time, duration_seconds)

                # Accumulate the time of the beat
                current_time += duration_seconds

        if measure.repeatClose > 0:
            print(f'Repeat Count {repeat_count}')
            # Set the (alternate repeat) jump to the next measure
            next_jump = current_measure + 1
            # Jump back to where the repeat begins
            current_measure = repeat_measure
            # Decrement the measure's repeat counter
            measure.repeatClose -= 1
            repeat_count += 1
        else:
            if measure.repeatClose == 0:
                print(f'Repeat Count {repeat_count}')
                repeat_count = 0
            # Increment the measure pointer
            current_measure += 1

    # Obtain the final tracked notes
    stacked_notes = note_tracker.get_stacked_notes()

    return stacked_notes
