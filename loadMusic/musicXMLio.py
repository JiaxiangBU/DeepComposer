import music21 as m21

def musicXMLtoArray(file_name):
    # In the current stage, we only consider piano song
    # So, there are two parts in music21.stream.Score
    # (left, right) hand
    mxml = m21.converter.parse(file_name)
    mparts = mxml.parts
    if len(mparts) != 2:
        print('The number of part is not 2')

    state_arr = list()

    for item in mparts:
        indi_arr = list()
        for jtem in item.flat.elements:
            tmp_type = type(jtem)
            if tmp_type == m21.note.Note:
                print(jtem.pitch.midi, jtem.duration.quarterLength)
            elif tmp_type == m21.note.Rest:
                print(-1, jtem.duration.quarterLength)
            elif tmp_type == m21.chord.Chord:
                pitches = list()
                for i in jtem.pitches:
                    pitches.append(i.midi)
                print(pitches, jtem.duration.quarterLength)
        break