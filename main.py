import string

from keras.preprocessing.text import Tokenizer
from numpy import array

from keras.utils import to_categorical

from keras.models import Sequential


from keras.layers import Dense,LSTM,Embedding

from pickle import load,dump

from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences


from random import randint





def loaded(file):

	file = open(file,"r")

	text = file.read()

	file.close()

	return text 


doc = loaded("republic_clean.txt")



def clean(doc):

	doc = doc.replace("--"," ")

	tokens = doc.split(" ")

	table = str.maketrans("","",string.punctuation)

	tokens = [i.translate(table) for i in tokens]

	tokens = [word.lower() for word in tokens if word.isalpha()]

	return tokens


tokens = clean(doc)

# print(tokens[:100])

print("Total tokens ",len(tokens))

print("Unique tokens",len(set(tokens)))


length = 50 + 1 # 50 input and 1 output


sequence = list()

for i in range(length,len(tokens)):
	seq = tokens[i-length:i]

	line = " ".join(seq)

	sequence.append(line)

print("Total sequence:",len(sequence))

def save_file(sequence,out_file):

	data = '\n'.join(sequence)

	file = open(out_file,"w")
	file.write(data)
	file.close


save_file(sequence,"republic_sequences.txt")



doc = loaded("republic_sequences.txt")

lines = doc.split("\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)

sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1


sequences = array(sequences)

#print(sequences.shape)  (96427, 51)

x, y = sequences[:,:-1], sequences[:,-1]

# print(x.shape,y.shape)  (96427, 50) (96427,)

y = to_categorical(y, num_classes=vocab_size) #converts to one-hot for each word

seq_length = x.shape[1] #50

print(seq_length) #50

print(x.shape,y.shape) #(96427, 50) (96427, 6578)


def model_creation():

	model = Sequential()

	model.add(Embedding(vocab_size,50,input_length=seq_length))

	model.add(LSTM(100,return_sequences=True))

	model.add(LSTM(100))

	model.add(Dense(100,activation="relu"))

	model.add(Dense(vocab_size,activation="softmax"))

	print(model.summary())



	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(x, y, batch_size=128, epochs=100)

	model.save('model.h5')

	dump(tokenizer, open('tokenizer.pkl', 'wb'))




def generate_seq(model,tokenizer,seq_length
	,seed_text,n_words):

	result = list()

	in_text = seed_text

	for _ in range(n_words):

		#converting the text to integers
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		#prediction
		yhat = model.predict_classes(encoded, verbose=0)
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)



doc = loaded('republic_sequences.txt')
lines = doc.split('\n')


seq_length = len(lines[0].split()) - 1



#loading the model 
model = load_model('model.h5')
 
tokenizer = load(open('tokenizer.pkl', 'rb'))
 
#choosing a random text
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')
 
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)