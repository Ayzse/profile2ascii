import numpy as np
import scipy as sp
import scipy.signal as sig
import itertools
import sklearn.neighbors as neighbors
import png

r = png.Reader('doug.png')
l = list(r.read()[2])

image2d = np.vstack(itertools.imap(np.uint16, l))
image3d = np.reshape(image2d, (651, 651, 3))

filter_edge = np.array([[0,0,0], [-1,1,0], [0,0,0]])

modified = []
for i in range(3):
	modified.append(sig.convolve2d(image3d[:,:,i], filter_edge, mode='valid'))

output = np.abs(np.reshape(np.dstack(modified), (649, (649)*3)))

f = open('file.png', 'wb')
w = png.Writer(649, 649)
w.write(f, output)
f.close()

r = png.Reader('ascii-dos.png')
l = list(r.read()[2])
asc = np.vstack(itertools.imap(np.uint16, l))
# make a training vector to hold the ascii values
train_data = np.zeros((12*8, 16*16))
for row in range(np.shape(asc)[0]):
	for column in range(np.shape(asc)[1]):
		train_data[(row%12)*8 + column%8, ((row/12)*16) + (column/8)/2] = asc[row, column]



"""
cutouts = np.zeros((96, 4374))
im = image3d[3:, 3:, :]
for i in range(np.shape(im)[0]):
	for j in range(np.shape(im)[1]):
		cutouts[((i%12)*8 + (j%8)), ((i/12)*(648/8)) + (j/8)] = im[i,j,0]


guesses = model.predict(np.transpose(cutouts))
out = np.zeros((648, 648))
for row in range(out.shape[0]):
	for column in range(out.shape[1]):
		out[row, column] = train_data[((row%12)*8) + column%8, guesses[((row/12)*(648/8)) + (column/8)%(648/8)]]

for row in range(out.shape[0]):
	for column in range(out.shape[1]):
		if out[row, column] != 0:
			out[row, column] = 255
out2 = np.copy(out)
out3 = np.copy(out)

x = np.dstack([out, out2, out3])
x = np.reshape(x, (648, 648*3))
w = png.Writer(648, 648)
f = open('final{}.png'.format('try'), 'wb')
w.write(f, x)
f.close()

"""


cutouts = np.zeros((96, 4374))
im = np.dstack(modified)[1:,1:,:]
for i in range(np.shape(im)[0]):
	for j in range(np.shape(im)[1]):
		cutouts[((i%12)*8 + (j%8)), ((i/12)*(648/8)) + (j/8)] = im[i,j,0]

for distance in range(1, 50):
	for i in range(train_data.shape[0]):
		for j in range(train_data.shape[1]):
			if train_data[i,j] != 0:
				train_data[i,j] = distance


	model = neighbors.KNeighborsClassifier(n_neighbors=1)
	model.fit(np.transpose(train_data), range(256))
	guesses = model.predict(np.transpose(cutouts))

	out = np.zeros((648, 648))
	for row in range(out.shape[0]):
		for column in range(out.shape[1]):
			out[row, column] = train_data[((row%12)*8) + column%8, guesses[((row/12)*(648/8)) + (column/8)%(648/8)]]


	for row in range(out.shape[0]):
		for column in range(out.shape[1]):
			if out[row, column] != 0:
				out[row, column] = 255

	out2 = np.copy(out)
	out3 = np.copy(out)

	x = np.dstack([out, out2, out3])
	x = np.reshape(x, (648, 648*3))
	w = png.Writer(648, 648)
	f = open('final{}.png'.format(distance), 'wb')
	w.write(f, x)
	f.close()

