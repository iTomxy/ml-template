import argparse

"""convert a string to title case
Example scenario:
    A title string copy from ICLR is all capital,
    I want to perttify it before pasting to my paper list file.
"""

parser = argparse.ArgumentParser()
parser.add_argument('text', type=str, nargs='+', default="", metavar="STR",
    help='the string to convert, w/o delimiter/quotation mark')
args = parser.parse_args()

ABBREVIATION = [
    "e.g.", "i.e.", "v.", "vs.", "v.s.", "w/", "w/o",
]
PREPOSITION = [
    "aboard", "about", "above", "across", "after", "against", "along", "amid", "among", "anti",
    "around", "as", "at", "before", "behind", "below", "beneath", "beside", "besides", "between",
    "beyond", "but", "by", "despite", "down", 'during', "except", "for", 'from', 'in',
    'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'over', 'per',
    'since', 'than', 'through', 'to', 'toward', 'towards', 'under', 'underneath', 'unlike', 'until',
    'up', 'upon', 'versus', 'via', 'with', 'within', 'without',
]
CONJUNCTION = [
    "after", "although", "and", "as", "because", "before", "besides", "but", "cuz", "either",
    "for", "how", "if", "neither", "nor", "once", "or", "since", "so", "that",
    "until" "when", "whether", "yet",
]
AUXILIARY_VERT = ["am", "is", "was", "were", "can", "could", "may", "might", "must", "ought", "shall", "should", "would"]
ARTICLE = ["a", "an", "the"]
# MISC = ["using", "based"]
MINOR_WORDS = set(ABBREVIATION + PREPOSITION + CONJUNCTION + AUXILIARY_VERT + ARTICLE)# + MISC)

is_first = True # first word of the title or after some delimiter
res = ""
for i, s in enumerate(args.text):
    # for s in ss.split():
    if not (s.isupper() and ':' == s[-1] and 0 == i): # not all capital
        s = s.lower()
        if is_first: # captialise no matter it is minor words or not
            s = s.capitalize()
            is_first = False
        elif s not in MINOR_WORDS:
            s = s.capitalize()

    res += s + ' '
    if not s[-1].isalnum(): # meets delimiter
        is_first = True

if ' ' == res[-1]:
    res = res[:-1] # rm last space
print(res)

