import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
 
# Loading doc into memory
def load_doc(filename):
	# Opening the file as read only
	file = open(filename, 'r')
	# Read all text
	text = file.read()
	# Closing the file
	file.close()
	return text

def normalization(name,accepted_chars):
	# Converting all characters of a name to lowercase and removing space and newline
  line = [i.lower() for i in name if i in accepted_chars]
  return line

def name_encoding(names,mapping,accepted_chars,vector_length):
	# Encoding every character in a name into integer
  encoded_names = []
  for name in names:
    encoded_name = normalization(name,accepted_chars)
    # Mapping each character to integer
    encoded_name = [mapping[char] for char in encoded_name]
    encoded_names.append(encoded_name)
  # Padding the encoded sequence to ensure uniformity of length
  encoded_names = pad_sequences(encoded_names,maxlen=vector_length,padding='post')
  
  return encoded_names

def model_building(hosp_file,name_file):
	# Reading the files and storing names in a list
	hosp_raw_text = load_doc(hosp_file)
	names_raw_text = load_doc(name_file)
	hosp_lines = hosp_raw_text.split('\n')
	names_lines = names_raw_text.split('\n')

	# Combining two lists into one
	full_names_list = hosp_lines + names_lines
	
	# Creating combined corpus
	raw_text = '\n'.join(full_names_list)
	raw_text = raw_text.lower()
	
	# Reducing corpus to unique set of characters(Lower Case for convenience)
	chars = sorted(list(set(raw_text)))
	
	# Reducing unique set of characters to acceptable set
	accepted_chars = [i for i in chars if i !=' ' and i != '\n']

	# Creating character to integer mapping 
	mapping = dict((c, i) for i, c in enumerate(accepted_chars))

	# Reducing all names to the acceptable character set to calculate input lenght
	full_names_list = [normalization(i,accepted_chars) for i in full_names_list]
	
	# Assigning labels(0 for person, 1 for hospital) 
	labels_hosp = np.ones((len(hosp_lines),1)) 
	labels_name = np.zeros((len(names_lines),1))
	
	# Calculating vector length and vocabulary size
	vector_length = len(max(full_names_list))
	vocab_size = len(mapping)

	#Encoding all names to digits
	encode_hospital = name_encoding(hosp_lines,mapping,accepted_chars,vector_length)
	encode_names = name_encoding(names_lines,mapping,accepted_chars,vector_length)
	
	#Appending labels to respective sets
	hospital_set = np.append(encode_hospital,labels_hosp,axis=1)
	names_set = np.append(encode_names,labels_name,axis=1)
	
	#Combining two sequence matrices into one and shuffling them randomly
	complete_names = np.append(hospital_set,names_set,axis=0)
	np.random.shuffle(complete_names)

	#Train, test and validation split(60%, 200% and 20% respectively) and
	#conversion of each dataset into one hot encoding
	train, test, validate = np.split(complete_names, [int(.6*len(complete_names)), int(.8*len(complete_names))])
	X_train, y_train = train[:,:-1], train[:,-1]
	sequences = [to_categorical(x, num_classes=vocab_size) for x in X_train]
	X_train = np.array(sequences)
	y_train = to_categorical(y_train, num_classes=2)
	
	X_validate, y_validate = validate[:,:-1], validate[:,-1]
	sequences = [to_categorical(x, num_classes=vocab_size) for x in X_validate]
	X_validate = np.array(sequences)
	y_validate = to_categorical(y_validate, num_classes=2)

	X_test, y_test = test[:,:-1], test[:,-1]
	sequences = [to_categorical(x, num_classes=vocab_size) for x in X_test]
	X_test = np.array(sequences)
	y_test = to_categorical(y_test, num_classes=2)

	#Calculating no. of hidden nodes
	hidden_nodes = int(2/3 * (vector_length * vocab_size))
	print("No. of hidden nodes are: ",hidden_nodes)

	#Training and validating the model
	print("Begin training")
	model = Sequential()
	model.add(LSTM(hidden_nodes,input_shape=(vector_length, vocab_size)))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	print(model.summary())
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	batch_size=1000
	model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_validate, y_validate))

	# Evaluation of the model via test set
	print("Evaluate on test data")
	results = model.evaluate(X_test, y_test)
	print("test loss, test acc:", results)

	return (model,mapping,accepted_chars,vocab_size,vector_length)

def main():
	hosp_file = './hospitals.txt'
	name_file = './names.txt'
	model,mapping,accepted_chars,vocab_size,vector_length = model_building(hosp_file,name_file)
	# Implementing the model via user input
	while(True):
		text = input("Enter name(Enter 'Exit' to quit): ")
		if text.lower() == 'exit':
			break
		encoded_text = name_encoding([text],mapping,accepted_chars,vector_length)
		one_hot_encoded_text = to_categorical(encoded_text,num_classes=vocab_size)
		result = model.predict_proba(one_hot_encoded_text)
		# Displaying each class along with respective probability values
		print("Human Name: {0},Hospital Name {1}".format(result[0][0],result[0][1]))

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
