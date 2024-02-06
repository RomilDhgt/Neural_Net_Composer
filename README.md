# Recurrent Neural Net Composer

Play the .mid files to see what NN Composer has created! 

The jupyter notebook is used to generate music from a Recurrent Neural Network. In the first segment of the notebook, the code prepares the data for training a neural network for music generation. It creates numerical mappings for pitchnames and sets the sequence length for input data. The input sequences and their corresponding outputs are then generated from the music data. In the second segment, the trained model is used to generate a sequence of musical notes. It selects a random starting point from the input data and iteratively predicts and decodes notes using the model. Lastly, the generated sequence is transformed into Note and Chord objects, with the offset adjusted to prevent note stacking, and saved as a MIDI file for playback. These segments collectively form the foundation for training and utilizing a generative musical AI.

