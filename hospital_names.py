import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences

# Load doc into memory
def load_doc(filename):
	# Opening the file as read only
	file = open(filename, 'r')
	# Reading all text
	text = file.read()
	# Closing the file
	file.close()
	return text

def prepare_seq(filename,length):
	# Loading raw text and print it
	raw_text = load_doc(filename)
	# Creating single list of all tokens(words) in text
	tokens = raw_text.split()
	raw_text = ' '.join(tokens)
	# Creating fixed length sequences of characters
	sequences = []
	for i in range(length, len(raw_text)):
		seq = raw_text[i-length:i+1]
		sequences.append(seq)
	print('Total Sequences: %d' % len(sequences))

	# Merging sequences into single string to return
	data = '\n'.join(sequences)
	return data
 
def prepare_data(filename,length):
	
	raw_text = prepare_seq(filename,length)
	lines = raw_text.split('\n')

	#Mapping each character in each sequence to an integer
	chars = sorted(list(set(raw_text)))
	mapping = dict((c, i) for i, c in enumerate(chars))
	sequences = list()
	#Encoding characters according to mapping sequence
	for line in lines:
		encoded_seq = [mapping[char] for char in line]
		sequences.append(encoded_seq)
	return (mapping,sequences)
 
def model_train(mapping,sequences):

	vocab_size = len(mapping)
	print('Vocabulary Size: %d' % vocab_size)
	
	# Converting matrix of encoded numbers to data and decision variable
	sequences = np.array(sequences)
	X, y = sequences[:,:-1], sequences[:,-1]
	
	# One hot encoding of categories in training data and decision variables
	sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
	X = np.array(sequences)
	y = to_categorical(y, num_classes=vocab_size)
	
	# Building the model and training
	model = Sequential()
	model.add(LSTM(85,input_shape=(X.shape[1], X.shape[2])))
	model.add(Dense(vocab_size, activation='softmax'))
	print(model.summary())
	
	# Fitting the model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X, y, epochs=100, verbose=2)

	return model

def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# Generating a fixed number of characters
	for _ in range(n_chars):
		# Encoding the characters as integers
		encoded = [mapping[char] for char in in_text]
		# Truncating sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# One hot encoding
		encoded = to_categorical(encoded, num_classes=len(mapping))
		# Predicting next character
		yhat = model.predict_classes(encoded, verbose=0)
		# Reversing map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# Appending to input
		in_text += char
	return in_text

def main():
	length = 20
	filename = "./hospitals.txt"
	mapping,sequences = prepare_data(filename,length)
	model = model_train(mapping,sequences)
	print("Starting prediction")
	while(True):
		# Entering inital seed text
		seed_text = input("Enter seed text(Type Exit to quit): ")
		if seed_text.lower() == 'exit':
			break
		in_text = generate_seq(model,mapping,length,seed_text,40)
		# Printing newly obtained hospital names
		print("Newly generated hospital names are:\n")
		print("in_text")

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()

 
