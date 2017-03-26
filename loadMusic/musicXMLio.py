import music21 as m21
import numpy as np


def musicXMLtoArray(file_name):
    # make note to lstm inputs
    # press the note       -> [note_position,1]
    # remaining the note -> [note_position,0]
    # end the note         -> [0,0]
    # In the current stage, we only consider piano song
    # So, there are two parts in music21.stream.Score
    # (left, right) hand

    lowest_note = 24
    highest_note = 102
    note_span = highest_note - lowest_note

    mxml = m21.converter.parse(file_name)
    mparts = mxml.parts

    if len(mparts) != 2:
        print('The number of part is not 2')

    state_arr = list()

    for item in mparts:
        indi_arr = list()
        for jtem in item.flat.elements:
            tmp_arr = np.zeros([2, note_span])
            tmp_type = type(jtem)

            if tmp_type == m21.note.Note:
                tmp_duration = int(jtem.duration.quarterLength * 8)

                tmp_arr[:, jtem.pitch.midi - lowest_note] = 1
                indi_arr.append(np.copy(tmp_arr))
                tmp_arr[1, jtem.pitch.midi - lowest_note] = 0
                for i in range(1, tmp_duration):
                    indi_arr.append(np.copy(tmp_arr))

            elif tmp_type == m21.note.Rest:
                tmp_duration = int(jtem.duration.quarterLength * 8)

                tmp_arr[:, :] = 0
                for i in range(tmp_duration):
                    indi_arr.append(np.copy(tmp_arr))

            elif tmp_type == m21.chord.Chord:
                tmp_duration = int(jtem.duration.quarterLength * 8)

                pitches = list()
                for i in jtem.pitches:
                    pitches.append(i.midi - lowest_note)
                pitches = np.array(pitches)

                tmp_arr[:, pitches] = 1
                indi_arr.append(np.copy(tmp_arr))
                tmp_arr[1, pitches] = 0
                for i in range(1, tmp_duration):
                    indi_arr.append(np.copy(tmp_arr))

        indi_arr = np.array(indi_arr).reshape([len(indi_arr), 2 * note_span])
        state_arr.append(indi_arr)

    return state_arr
