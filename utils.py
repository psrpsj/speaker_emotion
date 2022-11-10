import nltk

from tqdm import tqdm


def label_to_num(data):
    label_dict = {
        "neutral": 0,
        "joy": 1,
        "surprise": 2,
        "anger": 3,
        "sadness": 4,
        "disgust": 5,
        "fear": 6,
    }
    return data.map(label_dict)


def num_to_label(data):
    num_dict = {
        0: "neutral",
        1: "joy",
        2: "surprise",
        3: "anger",
        4: "sadness",
        5: "disgust",
        6: "fear",
    }
    return data.map(num_dict)


def stopword(data):
    nltk.download()
    stop_word = set(nltk.corpus.stopwords.word("english"))
    for idx in tqdm(range(len(data))):
        line = data["Utterance"][idx]
        line_token = nltk.tokenize.word_tokenize(line)
        result = []
        for word in line_token:
            if word not in stop_word:
                result.append(word)
        data["Utterance"][idx] = " ".join(result)
    return data
