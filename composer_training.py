import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint

# The notes array has to be filled with the data in the music_data folder
notes = []

# For each file in the music_data folder, the music21 converter will load the .mid music files into the midi variable 
for file in glob.glob("music_data/*.mid"):
    midi = converter.parse(file)
    parsed_notes = None
    
    # Parse based on instruments, if not instruments then file has a flat structure 
    try:
        s2 = instrument.partitionByInstrument(midi)
        parsed_notes = s2.parts[0].recurse() 
    except:
        parsed_notes = midi.flat.notes
    
    # Populating the notes array with two types of notes, note or chord
    for element in parsed_notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

#with open('data/notes','wb') as filepath:
#    pickle.dump(notes, filepath)

num = len(set(notes))

sequence_len = 100

pitchnames = sorted(set(item for item in notes))

note_in_num = dict((note, number) for number, note in enumerate(pitchnames))

net_input = []
net_output = []

for i in range(0, len(notes) - sequence_len, 1):
    seq_in = notes[i:i + sequence_len]
    seq_out = notes[i+sequence_len]
    net_input.append([note_in_num[char] for char in seq_in])
    net_output.append([note_in_num[seq_out]])

num_patterns = len(net_input)

net_input = numpy.reshape(net_input, (num_patterns, sequence_len, 1))

net_input = net_input / float(num)

net_output = to_categorical(net_output)

model = Sequential
model.add(LSTM(
    512,
    input_shape=(net_input.shape[1], net_input.shape[2]),
    recurrent_dropout=0.3,
    return_sequences=True
))
model.add(LSTM(
    512,
    return_sequences=True,
    recurrent_dropout=0.3
))
model.add(LSTM(512))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(num))
model.add(Activation('softmax'))
model.compile(loss='catergorical_crossentropy', optimizer='rmsprop')

checkpoint = ModelCheckpoint(
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5",
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

callbacks_list = [checkpoint]

model.fit(net_input, net_output, epochs=200, batch_size=128, callbacks=callbacks_list)

