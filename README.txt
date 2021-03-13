## NOTE: demo iqynb notebook is inside the folder called "demo" for simpler importing of demo-needed modules 

Clip.py
> A data type we used to store clips. Contains timestamp, chats, and its category. 

Utilities.py 
> Utility methods to be used across the entire project.

Clipper.py 
> A module that clips (clip objects) from json files downloaded using TwitchDownloader based on chat speed. 

Labeler.py 
> A terminal based module that lets the user label outputs from clipper. 

Tokenizer_kit.py
> A collection of methods of string processing used during the tokenization process.

ListOfStrings.py
> Creates a list of chat messages from stream json file. Each chat message is a list of strings. (Not used, moved module’s function to embedding.py).

Embedding.py
> Uses Gensim’s word2vec algorithm to train a word embedding using the words in a streamer’s chat.

Data_loader.py
> A collection of methods for each for loading and formatting data into learner friendly structures. 

Data_converter.py
> A collection of methods to convert loaded data into desired format (e.g. to suit the learner, or suitable for printing)

Learners:
> These modules take in data and train learners based on the data, printing accuracies. 
  Log_regression.py
  > Log regression model.

  MLP.py
  > Multi layered perceptron model. 

  RNN.py
  > Recurrent neural network model. 

  GRU.py
  > Gated recurrent unit model.

When_to_watch.py
> Take a .pkl file of labeled clips, print out all the interesting moments of a video suggested by the .pkl file.

ClipperV2.py
> Clip from json files downloaded from TwitchDownloader. But uses embedding and emotional similarities of chats within a time interval. 

miscellaneous.txt
> text file we used to jot down notes and things we’ve noticed while testing and implementing code. 

rnn.pt 
> pytorch file for the best rnn learner we have. 

foo.py 
> a module to test codes before practical use in one of the modules. 

Folders: 
  demo
  > folder which holds demonstration notebook and needed data. 

  > These folders are mainly used to hold json or pkl files of clips and other data. 

  Chatjsonfiles 
  > Folder to store json files downloaded from Twitch Downloader. 

  Clip_data
  > Folder that contains pkl files that store the results of clipping the data (timestamps, text, etc.) Each sub-folder in this folder contains a set of clips from a single streamer.

  Labeled_clip_data 
  > Contains pkl files that store the manual labels of each clip, in addition to timestamps, texts, etc.

  Chat_words 
  > output from ListOfStrings.py, intended for embedding. Not used because embedding does the entire streamline. 

  Word_vectors
  > Folder to store .kv file of embeddings. 

  Mislabeled
  > Folder to store mislabeled clips from learner output. 

  Model_labeled_result
  > clips that are labeled by our learners. Used to test if our learners are doing a good job labeling random twitch videos.

  Teo_chrono: 
  > folder to store clipperV2 input data. “Teo” is the streamer’s name. 
