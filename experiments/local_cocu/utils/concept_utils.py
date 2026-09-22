import re


STOP_WORDS={
"a",
"the",
"an",
"of",
"on",
"with",
"and",
"in",
"to"
}


def extract_words(caption):

    words=re.findall(
        r"[a-z]+",
        caption.lower()
    )

    return [
        w
        for w in words
        if w not in STOP_WORDS
    ]



def normalize_gallery(gallery):

    concepts=[]


    # 情况1:
    # ["dog","cat"]

    if isinstance(gallery,list):

        for x in gallery:

            if isinstance(x,str):
                concepts.append(x)


            elif isinstance(x,dict):

                if "concept" in x:
                    concepts.append(
                        x["concept"]
                    )


    # 情况2:
    # {"concept":[...]}

    elif isinstance(gallery,dict):

        if "concept" in gallery:

            concepts.extend(
                gallery["concept"]
            )


        else:

            for k in gallery.keys():

                concepts.append(k)



    return set(concepts)



def match_gallery(words,gallery):

    gallery=normalize_gallery(
        gallery
    )

    return [
        w
        for w in words
        if w in gallery
    ]
