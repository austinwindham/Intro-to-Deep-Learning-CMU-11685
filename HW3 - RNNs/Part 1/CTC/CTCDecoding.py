import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        y_probs = np.squeeze(y_probs, axis=2)  # shape: (len(symbols) + 1, seq_len)

        prev_symbol = None

        for t in range(y_probs.shape[1]):
            probs_t = y_probs[:, t]
            max_index = np.argmax(probs_t)
            max_prob = probs_t[max_index]
            path_prob *= max_prob

            # collapse and remove blanks
            if max_index != blank and max_index != prev_symbol:
                decoded_path.append(self.symbol_set[max_index -1])  

            prev_symbol = max_index

        decoded_path = "".join(decoded_path)

        return decoded_path, path_prob
        


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

#         Initialize instance variables

#         Argument(s)
#         -----------

#         symbol_set [list[str]]:
#             all the symbols (the vocabulary without blank)

#         beam_width [int]:
#             beam width for selecting top-k hypotheses for expansion

#         """


        self.symbol_set = ["-"] + symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = {"-": 1.0}, 0
        #initializations
        tempPaths = {}
        
        for t in range(T):

            currProb = y_probs[:, t, 0]
            bestPath = dict(sorted(bestPath.items(), key=lambda prob: prob[1])[-self.beam_width:])

            for path, score in bestPath.items(): # loop path
                for n, prob in enumerate(currProb): # loop symbols
                    
                    symbol = self.symbol_set[n]
                    newProb = prob*score
                    newPath = path 
                    
                    if path[-1] == "-": 
                        newPath = path[:-1] + symbol

                    elif symbol != path[-1] and not (t == T-1 and symbol=='-'):
                        newPath += symbol
                    
                    if newPath in tempPaths:
                        tempPaths[newPath] += newProb

                    else:
                        tempPaths[newPath] = newProb
                    
            bestPath = tempPaths
            tempPaths = {}
        
        best = max(bestPath, key=bestPath.get)

        merged_path_scores = { path[:-1] if path[-1] =="-" else path: prob for path, prob in bestPath.items() }
        
        return best, merged_path_scores