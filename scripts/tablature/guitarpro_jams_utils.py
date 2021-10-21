# Author: Jonathan Driedger (Chordify)
# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import amt_tools.tools as tools

# Regular imports
import numpy as np
import guitarpro
import librosa
import jams
import os

# TODO - clean this file up

QUARTER_LENGTH_TICKS = 960
STANDARD_TUNING = np.array([64, 59, 55, 50, 45, 40])
FORBIDDEN_CHARACTERS = ['/', '*', '|', '\"']

class Note:

    def __init__(self,fret,string,time,duration,):
        self.fret = fret
        self.string = string
        self.time = time
        self.duration = duration
        self.pitch = STANDARD_TUNING[self.string] + self.fret

    def add_duration(self, duration):
        self.duration += duration

class NoteTracker:

    def __init__(self,tempo):
        self.tempo = tempo
        self.beat_len_ticks = None
        self.finished_notes_per_string = [[],[],[],[],[],[]]
        self.last_notes = [None,None,None,None,None,None]

    def set_beat_len_ticks(self, beat_len_ticks):
        self.beat_len_ticks = beat_len_ticks

    def convert_ticks_to_seconds(self,ticks):
        return ticks / self.beat_len_ticks /self.tempo * 60

    def add_note(self, gp_note, time_ticks, duration_ticks):
        fret = gp_note.value
        string = gp_note.string - 1
        type = gp_note.type.name # for normal notes, name is "normal" and value 1

        time_seconds = self.convert_ticks_to_seconds(time_ticks)
        duration_seconds = self.convert_ticks_to_seconds(duration_ticks)
        note = Note(string=string, fret=fret,time=time_seconds,duration=duration_seconds)

        if self.last_notes[string] is None:
            self.last_notes[string] = note
        else:
            if type == "normal":
                self.finished_notes_per_string[string].append(self.last_notes[string])
                self.last_notes[string] = note
            else:
                self.last_notes[string].add_duration(duration_seconds)

    def get_final_notes_per_string(self):
        for n in self.last_notes:
            if n is not None:
                self.finished_notes_per_string[n.string].append(n)

        return self.finished_notes_per_string


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


def get_all_notes_from_track(track, tempo):
#def extract_stacked_notes_gpro_track(grpo_track, tempo):
    """
    Extract MIDI notes spread across strings within a GuitarPro track into a dictionary.

    Parameters
    ----------
    gpro_track : Track object (PyGuitarPro)
      GuitarPro track data
    tempo : int
      Track tempo for inferring note onset and duration

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    note_tracker = NoteTracker(tempo)

    measures = track.measures
    for m in measures:
        time_signature = (m.timeSignature.numerator,m.timeSignature.denominator.value)
        beat_len_ticks = 960 / (time_signature[1] / 4)
        note_tracker.set_beat_len_ticks(beat_len_ticks)

        measure_end_ticks = m.end

        voices = m.voices
        for v in voices:
            beats = v.beats
            for i,b in enumerate(beats):
                notes = b.notes
                if len(notes) == 0:
                    continue

                curr_beat_start_time_ticks = b.start
                if i < len(beats)-1:
                    curr_beat_end_time_ticks = beats[i+1].start
                else:
                    curr_beat_end_time_ticks = measure_end_ticks
                curr_beat_duration_ticks = curr_beat_end_time_ticks - curr_beat_start_time_ticks

                for n in notes:
                    note_tracker.add_note(n,time_ticks=curr_beat_start_time_ticks, duration_ticks=curr_beat_duration_ticks)

    return note_tracker.get_final_notes_per_string()


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
            ann.append(time=n.time, duration=n.duration, value=n.pitch)
            if n.time + n.duration > duration_track_string:
                duration_track_string = n.time + n.duration

        # ann.duration = duration_track_string # use thiis if each string annotation should be labeled with the duration of the individual string

        jam.annotations.append(ann)

        if duration_track_string > duration_track:
            duration_track = duration_track_string

    jam.file_metadata.duration = duration_track
    for ann in jam.annotations:
        ann.duration = duration_track

    jam.save(path_out)
