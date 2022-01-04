# Author: Jonathan Driedger (Chordify)
# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

# Regular imports
import numpy as np
import librosa
import jams

TICKS_PER_QUARTER_NOTE = 960
NOTE_TYPE_ENUM_REST    = 'rest'
NOTE_TYPE_ENUM_NORMAL  = 'normal'
NOTE_TYPE_ENUM_TIE     = 'tie'
NOTE_TYPE_ENUM_DEAD    = 'dead'


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

        self.string = string
        self.fret = fret
        self.onset = onset
        self.duration = duration

    def get_absolute_pitch(self, tuning=None):
        """
        Determine the absolute MIDI pitch of the note.
        TODO - is this function necessary? is it used for anything?

        Parameters
        ----------
        tuning : list or ndarray
            MIDI pitch of each open-string

        Returns
        ----------
        absolute_pitch : int
          MIDI pitch of the note
        """

        if tuning is None:
            # Default the guitar tuning
            tuning = librosa.note_to_midi(tools.DEFAULT_GUITAR_TUNING)

        # Add the fret to the open-string's MIDI pitch
        absolute_pitch = tuning[self.string] + self.fret

        return absolute_pitch

    def extend_note(self, duration):
        """
        Extend the note by a specified amount of time.

        Parameters
        ----------
        duration : float
          Amount of time to extend the note
        """

        self.duration += duration


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

    def track_note(self, gpro_note):
        """
        Update the currently tracked tempo.

        Parameters
        ----------
        gpro_note : PyGuitarPro Note object
          GuitarPro note information
        """

        # Extraction all relevant note information
        string_idx, fret, type = gpro_note.string - 1, gpro_note.value, gpro_note.type.name

        # Extract the timing information of the beat division
        # TODO - subtract 1 from duration to avoid frame overlap on boundary?
        onset_tick, duration_ticks = gpro_note.beat.start, gpro_note.beat.duration.time

        # Convert the note onset and duration from ticks to seconds
        onset_seconds = ticks_to_seconds(onset_tick, self.current_tempo)
        duration_seconds = ticks_to_seconds(duration_ticks, self.current_tempo)

        # TODO - remove this if it never happens
        if gpro_note.durationPercent != 1.0:
            print()

        # Scale the duration by the duration percentage
        duration_seconds *= gpro_note.durationPercent

        # Create a note object to keep track of the GuitarPro note
        note = Note(fret, onset_seconds, duration_seconds, string_idx)

        # Get the key corresponding to the string index
        key = self.string_keys[string_idx]

        if type == NOTE_TYPE_ENUM_NORMAL:
            # Add the new note to the dictionary under the respective string
            self.stacked_gpro_notes[key].append(note)
        elif type == NOTE_TYPE_ENUM_TIE:
            # Determine the last note that occurred on the string
            last_gpro_note = self.stacked_gpro_notes[key][-1]# \
                             #if len(self.stacked_gpro_notes[key]) else None <- should never happen if we get a tie
            # Extend the previous note by the current beat's duration
            last_gpro_note.extend_note(duration_seconds)
            self.stacked_gpro_notes[key][-1] = last_gpro_note
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
        for key in self.string_keys:
            # Initialize empty arrays to hold note contents
            pitches, intervals = np.empty(0), np.empty((0, 2))

            # Loop through all the GuitarPro notes for the string
            for note in self.stacked_gpro_notes[key]:
                # Add the absolute pitch of the note
                pitches = np.append(pitches, librosa.note_to_midi(key) + note.fret)
                # Add the onset and offset of the note
                intervals = np.append(intervals, [[note.onset, note.onset + note.duration]], axis=0)

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

    """"""
    # TODO - remove when there is a check for zero notes in outer script
    all_notes = []
    measures = gpro_track.measures
    for m in measures:
        voices = m.voices
        for v in voices:
            beats = v.beats
            for i, b in enumerate(beats):
                notes = b.notes
                all_notes += notes
    valid = valid and (len(all_notes) > 0)
    """"""

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

    # Loop through the track's measures
    # TODO - while loop to deal with repeats (invalidates method for obtaining onset and duration)?
    for n, measure in enumerate(gpro_track.measures):
        # Loop through voices within the measure
        for voice in measure.voices:
            # Loop through the beat divisions of the measure
            for i, beat in enumerate(voice.beats):
                # Check if there are notes to process
                if len(beat.notes) == 0:
                    continue

                # Check if there are any tempo changes
                if beat.effect.mixTableChange is not None:
                    # TODO - remove this later
                    print(n, beat.effect.mixTableChange)

                    # Extract the updated tempo
                    new_tempo = beat.effect.mixTableChange.tempo.value
                    # Update the tempo of the note tracker
                    note_tracker.set_current_tempo(new_tempo)

                    # TODO - remove this if it never happens
                    if beat.effect.mixTableChange.tempo.duration != 0:
                        print()

                # Loop through the notes in the beat division
                for note in beat.notes:
                    # TODO - how to deal with letRing, staccato, and other NoteEffects?
                    note_tracker.track_note(note)

    # Obtain the final tracked notes
    stacked_notes = note_tracker.get_stacked_notes()

    return stacked_notes


def convert_notes_per_string_to_jams(notes_per_string, path_out):
#def write_stacked_notes_jams(stacked_notes, jams_path):
    """
    Helper function to create a JAMS file and populate it with stacked notes.

    Parameters
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    jams_path : string
      Path to JAMS file to write
    """

    jam = jams.JAMS()
    duration_track = 0
    for string in np.arange(5,-1,-1):
        # for string in range(6):
        curr_notes = notes_per_string[string]

        # ann = jams.Annotation(namespace='note_midi', time=0, duration=jam.file_metadata.duration)
        ann = jams.Annotation(namespace='note_midi', time=0, duration=0)
        ann.annotation_metadata = jams.AnnotationMetadata(data_source=5-string) # note: In JAMS the string with index 0 is the low e-string, in GP it is the high e string

        duration_track_string = 0
        for n in curr_notes:
            ann.append(time=n.onset, duration=n.duration, value=n.get_absolute_pitch())
            if n.onset + n.duration > duration_track_string:
                duration_track_string = n.onset + n.duration

        # ann.duration = duration_track_string # use thiis if each string annotation should be labeled with the duration of the individual string

        jam.annotations.append(ann)

        if duration_track_string > duration_track:
            duration_track = duration_track_string

    jam.file_metadata.duration = duration_track
    for ann in jam.annotations:
        ann.duration = duration_track

    jam.save(path_out)
