def filter_text(generated_move):
    # removes any moves that follow after the genrated move
    # needs to be updated to remove any move after the first that says "This move is..."
    # this will prevent moves that are generated after the seed
    # filter text such that additional moves are not used.
    # takes care of potential tokenizing problems

    generated_move = generated_move.replace("Sp.", "Special")
    sentences = generated_move.split(".")

    if len(sentences) > 2:
        #check if multiple sentences start with "This move"
        # remove sentences that describe the second move
        #ret_set = ". ".join(sentences[:2])
        this_move_indexes = [0]
        for idx, sent in enumerate(sentences):
            if idx > 0:
                if "this move is called" in sent.lower():
                    this_move_indexes.append(idx)
        # if this_move_indexes is longer than 1, then filter.
        if len(this_move_indexes) > 1:
            #filter to the second index, exclusive
            sentences = sentences[:this_move_indexes[1]]
    ret_set = "\n".join(sentences)
    return ret_set


def format_moves(moves):
    # given a list of dictionaries of moves
    # formats into a string with newlines
    move = filter_text(moves[0]["generated_text"])
    return move
