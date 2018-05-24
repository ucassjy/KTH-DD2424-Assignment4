def ReadAndProcess(datafile):
    with open(datafile, 'r') as f:
        l = f.read()
    print(len(l))
    print(l[0])
    char = list(set(l))
    char.sort()
    ind = [i+1 for i in range(len(char))]
    char_to_ind = dict(zip(char, ind))
    ind_to_char = dict(zip(ind ,char))
    return char_to_ind, ind_to_char

char_to_ind, ind_to_char = ReadAndProcess('goblet_book.txt')
