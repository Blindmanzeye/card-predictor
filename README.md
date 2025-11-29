I train and tweak an existing neural network to be able to read a png that contains an image of a card.
The model will output a list of float values between 0 and 1 that dictate it's confidence that the image
is a certain card. These values are then displayed in a bar graph, as well as the png image, using mathplotlib
