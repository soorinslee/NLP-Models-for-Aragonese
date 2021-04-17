import random

# create masked version of train dataset
with open('encodedTrain.csv', 'r') as file:
    with open('maskedTrain.csv', 'w') as masked:
        for row in file:
            if len(row.split()) > 0:
                i = random.randint(0, len(row.split()) - 1)
                count = 0
                sentence = ''

                for word in row.split():
                    if i != count:
                        sentence = sentence + ' ' + word
                    else:
                        sentence = sentence + ' [MASK]'
                    count = count + 1

                masked.write(sentence)
                masked.write('\n')
            else:
                masked.write(' ')
                masked.write('\n')