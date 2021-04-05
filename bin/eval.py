import os.path as osp
import spacy
import click


COLORS = [
    "red",
    "green",
    "blue",
    "black",
    "white",
    "cyan",
    "gray",
    "grey",
    "purple",
    "brown",
    "yellow",
    "pink",
    "magenta",
    "violet",
    "orange",
    "bright",
    "brighter",
    "brightest",
    "dark",
    "darker",
    "darkest",
    "blond",
    "gold",
    "silver",
    "beige",
]


POSITIONS = [
    'above',
     'away',
     'back',
     'behind',
     'below',
     'beside',
     'between',
     'bottom',
     'center',
     'close',
     'closest',
     'corner',
     'down',
     'end',
     'facing',
     'far',
     'farthest',
     'first',
     'front',
     'furthest',
     'last',
     'left',
     'leftmost',
     'lower',
     'meter',
     'middle',
     'near',
     'nearest',
     'next',
     'over',
     'right',
     'rightmost',
     'second',
     'side',
     'third',
     'top',
     'toward',
     'towards',
     'under',
     'up',
     'upper',
]



def read_file(filepath):
    nlp = spacy.load("en_core_web_sm")

    filepath = osp.abspath(osp.expanduser(filepath))
    with open(filepath, "r") as f:
        lines = [line.strip().split(",") for line in f.readlines()]

    data = []
    for line in lines:
        idx, split, text, I, U, IoU = line
        doc = nlp(text.replace("<unk>", "unk"))
        I = int(float(I))
        U = int(float(U))
        IoU = float(IoU)
        data.append([int(idx), split, doc, I, U, IoU])
    return data


def eval(data):
    total_I = total_U = 0.0
    count = 0
    for x in data:
        doc, I, U = x[2:5]
        total_I += I
        total_U += U
        count += 1
    if count == 0:
        return 0.0, 0
    IoU = round(100 * total_I / total_U, 2)
    return IoU, count


def contains_color(doc):
    for t in doc:
        if t.text in COLORS:
            return True
    return False


def not_contains_color(doc):
    return not contains_color(doc)


def contains_position(doc):
    for t in doc:
        if t.text in POSITIONS:
            return True
    return False


def not_contains_position(doc):
    return not contains_position(doc)


def contains_double_noun(doc):
    tags = [x.tag_ for x in doc]
    for i in range(len(tags)-1):
        curr, next = tags[i], tags[i+1]
        if curr.startswith("NN") and next.startswith("NN"):
            return True
    return False


def not_contains_double_noun(doc):
    return not contains_double_noun(doc)


def contains_verb(doc):
    for t in doc:
        if t.tag_.startswith("VB"):
            return True
    return False


def contains_adjective(doc):
    for t in doc:
        if t.tag_.startswith("JJ"):
            return True
    return False


def contains_comparative_adjective(doc):
    for t in doc:
        if t.tag_ == "JJR":
            return True
    return False


def contains_superlative_adjective(doc):
    for t in doc:
        if t.tag_ == "JJS":
            return True
    return False


def contains_comparative_superlative_adjective(doc):
    for t in doc:
        if t.tag_ == "JJS" or t.tag_ == "JJR":
            return True
    return False


def contains_adverb(doc):
    for t in doc:
        if t.tag_.startswith("RB"):
            return True
    return False


def contains_comparative_adverb(doc):
    for t in doc:
        if t.tag_ == "RBR":
            return True
    return False


def contains_superlative_adverb(doc):
    for t in doc:
        if t.tag_ == "RBS":
            return True
    return False


def contains_comparative_superlative_adverb(doc):
    for t in doc:
        if t.tag_ == "RBS" or t.tag_ == "RBR":
            return True
    return False


def contains_multiple_adjectives(doc):
    for chunk in doc.noun_chunks: 
        count = 0
        for word in chunk:
            if word.tag_.startswith("JJ"):
                count += 1
        if count >= 2:
            return True
    return False


def contains_preposition(doc):
    for t in doc:
        if t.tag_ == "IN":
            return True
    return False


def contains_multiple_noun_chunks(doc):
    return len(list(doc.noun_chunks)) > 1


def excludes_color_includes_position(doc):
    return not contains_color(doc) and contains_position(doc)


def excludes_color_excludes_position(doc):
    return not contains_color(doc) and not contains_position(doc)


FUNCTIONS = {
    "color": contains_color,
    "-color": not_contains_color,
    "pos": contains_position,
    "-pos": not_contains_position,
    "IN": contains_preposition,
    "JJ*": contains_adjective,
    "JJ+": contains_comparative_superlative_adjective,
    "JJR": contains_comparative_adjective,
    "JJS": contains_superlative_adjective,
    "n-JJ*": contains_multiple_adjectives,
    "RB*": contains_adverb,
    "RB+": contains_comparative_superlative_adverb,
    "RBR": contains_comparative_adverb,
    "RBS": contains_superlative_adverb,
    "NN+NN": contains_double_noun,
    "n-NN": contains_multiple_noun_chunks,
    "VB*": contains_verb,
    # "-color+pos": excludes_color_includes_position,
    # "-color-pos": excludes_color_excludes_position,
}


@click.command()
@click.argument("filepath", required=True, type=click.Path(exists=True))
@click.option('--exclude-colors', is_flag=True)
def main(filepath, exclude_colors):
    data = read_file(filepath)
    if exclude_colors:
        data = [x for x in data if not_contains_color(x[2])]
    results = {}
    for (cat, func) in FUNCTIONS.items():
        IoU, count = eval([x for x in data if func(x[2])])
        results[f"{cat} ({count})"] = str(IoU)
    results = sorted(results.items(), key=lambda x: x[0].lower())
    print(",".join([x[0] for x in results]))
    print(",".join([x[1] for x in results]))


if __name__ == "__main__":
    main()