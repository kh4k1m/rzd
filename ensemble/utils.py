
def get_palette():
    labels = ['gound', 'rails', 'main_rail', 'train']
    colors = [(0, 0, 0), (6, 6, 6), (7, 7, 7), (10, 10, 10)]
    id2color = {id: color for id, color in enumerate(colors)}
    label2id = {label: id for id, label in enumerate(labels)}
    id2label = {id: label for id, label in enumerate(labels)}
    return id2label, label2id, id2color
