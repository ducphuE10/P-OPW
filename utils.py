def get_data(path):
    with open(path, 'r') as f:
        data = []
        labels = []
        for line in f.readlines():
            label = int(float(line.split()[0]))
            record = [float(j) for j in  line.split()[1:]]
            data.append(record)
            labels.append(label)

    return data, labels

train_data, train_labels = get_data('FacesUCR/FacesUCR_TRAIN.txt')
