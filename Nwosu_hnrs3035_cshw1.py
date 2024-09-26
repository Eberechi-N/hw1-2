import re
import argparse
import pickle
import random
from collections import Counter, defaultdict



class NGramModel:

    def __init__(self, n):
        self.n = n
        self.vocabulary = set()
        self.ngram_counts = {}

#look over probabilty and change code

    def train(self, corpus):
        # Split the corpus into words and punctuation
        words = re.findall(r"\w+|[^\w\s]", corpus)
        #https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
        self.vocabulary = set(words)
        print(f"Vocabulary: {self.vocabulary}")

        # Track which words follow each unique word or word-pair
        for i in range(len(words) - self.n):
            ngram = tuple(words[i:i + self.n])  # n-gram (1-word or 2-word tuple)
            next_word = words[i + self.n]  # The word that follows

            if ngram not in self.ngram_counts:
               self.ngram_counts[ngram] = []
            self.ngram_counts[ngram].append(next_word)
            #https://github.com/npapernot/bigram-next-word-predictor/blob/master/bigram-next-word-predictor.py

        # Calculate probabilities
        for ngram, next_words in self.ngram_counts.items():
            total_count = len(next_words)
            word_counts = {w: next_words.count(w) for w in set(next_words)}
            self.ngram_counts[ngram] = {
                w: count / total_count
                for w, count in word_counts.items()
            }
        print(f"N-gram counts: {self.ngram_counts}")

        #https://github.com/khanhnguyendata/ngram/blob/master/analysis/unigram.py


    def predict_next_word(self, input, deterministic=False):
        # Check if input n-gram exists in the model
        if input not in self.ngram_counts:
            print(
                f"Error: The input n-gram '{input}' does not exist in the model's vocabulary."
            )
        exit

        # Get the probability distribution of the next word
        next_word_probs = self.ngram_counts[input]

        if deterministic:
            # Return the word with the highest probability
            next_word = max(next_word_probs, key=next_word_probs.get)
        else:
            # Sample the next word based on the probability distribution
            words = list(next_word_probs.keys())
            probabilities = list(next_word_probs.values())
            next_word = random.choices(words, weights=probabilities, k=1)[0]

        return next_word



class BPE:
    def __init__(self):
        """
        Initializes the BPE object with an empty vocabulary and a counter for token IDs.
        """
        self.vocabulary = {}
        self.token_id_counter = 1

    def train(self, data, k=500):
        """
        Train the BPE model on the given data, performing a specified number of merges.
        """
        # Convert the text to lowercase and strip any extra spaces
        data = data.lower()
        text_tokens = list(data)

        # Function to extract consecutive token pairs from the current list of tokens
        def get_pairs(tokens):
            """
            Extract all consecutive token pairs from the current list of tokens.
            """
            the_pairs = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                the_pairs.append(pair)
            return the_pairs

        # Function to replace a frequent token pair with a new merged token
        def replace_pair(tokens, max_pair):
            """
            Replace the most frequent pair in the text with a new merged token.
            This function now pairs characters next to each other without skipping tokens.
            """
            new_token = ''.join(max_pair)  # Create a new token by merging the pair
            new_tokens = []  # Initialize the list to store updated tokens
            i = 0  # Index for iterating over tokens

            while i < len(tokens) - 1:
                # Check if the current and next tokens form the pair to replace
                if tokens[i] == max_pair[0] and tokens[i + 1] == max_pair[1]:
                    new_tokens.append(new_token)  # Add the new merged token
                    i += 2  # Move past the merged pair
                else:
                    new_tokens.append(tokens[i])  # Add the current token to the new list
                    i += 1  # Move to the next token

            # Append the last token if it is not part of a pair
            if i < len(tokens):
                new_tokens.append(tokens[-1])

            return new_tokens


        # Perform BPE merges for the specified number of iterations
        for merge in range(k):
            # Get all token pairs in the current token list
            pairs = get_pairs(text_tokens)

            # Count occurrences of each pair
            pair_counts = Counter(pairs)  

            if not pair_counts:
                break  # Exit if no more pairs are found

            # Find the most frequent pair
            max_pair = pair_counts.most_common(1)[0][0]
            text_tokens = replace_pair(text_tokens, max_pair)  
            new_token = ''.join(max_pair) 
            if new_token not in self.vocabulary:
                self.vocabulary[new_token] = self.token_id_counter
                self.token_id_counter += 1

        return self.vocabulary  # Return the final vocabulary after all merges

    def tokenize(self, text):
        """
        Tokenizes the given text using the trained vocabulary.

        Args:
            text: The text to tokenize.

        Returns:
            A tuple containing:
                - The tokens as a list.
                - The token IDs as a list.
        """
        # 1. Convert the text to lowercase and split into characters
        tokens = list(text.lower())

        # 2. Iteratively apply the vocabulary merges, in the order they were learned
        for token in self.vocabulary.keys():
            merged_token = token
            i = 0
            # Try to replace occurrences of the token (symbol) in the text
            while i < len(tokens) - 1:
                # Check if we have a match for the token in the list
                if ''.join(tokens[i:i + len(merged_token)]) == merged_token:
                    # Replace the matched characters with the merged token
                    tokens[i:i + len(merged_token)] = [merged_token]
                i += 1

        # 3. Map tokens to their corresponding IDs from the vocabulary
        token_ids = [self.vocabulary[token] for token in tokens if token in self.vocabulary]

        return tokens, token_ids







#need to remove loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "activity",
        choices=["train_ngram", "predict_ngram", "train_bpe", "tokenize", "loop", "test"],
                help= 'these are the activities that can run!'
                )
    parser.add_argument("--data", help="Path to the training data corpus")
    parser.add_argument("--save", help="Path to save the trained model")
    parser.add_argument("--load", help="Path to load the trained model")
    parser.add_argument("--word", help="First word(s) for prediction")
    parser.add_argument("--nwords",
                        type=int,
                        help="Number of words to predict")
    parser.add_argument("--text", help="Text to tokenize")
    parser.add_argument("--n",
                        type=int,
                        choices=[1, 2],
                        help="Order of the ngram (1 for unigram, 2 for bigram)")
    parser.add_argument("--d",
                        action="store_true",
                        help="Use deterministic prediction")

    args = parser.parse_args()

    if args.activity == "train_ngram":
        if not args.data or not args.n:
            print("Error: --data and --n are required for train_ngram")
            return
        try:
            with open(args.data, 'r', encoding='utf-8') as file:
                data = file.read()
            model = NGramModel(args.n)
            model.train(data)
            if args.save:
                with open(args.save, 'wb') as file:
                    pickle.dump(model, file)
            print("NGram model trained and saved.")
        except:
                print("The file was unable to open. File might not exist. ")

    elif args.activity == "predict_ngram":
        if not args.load or not args.word or not args.nwords:
            print(
                "Error: --load, --word, and --nwords are required for predict_ngram"
            )
            return
        try:
            with open(args.load, 'rb') as file:
                    model = pickle.load(file)
            words = args.word.split()
                # Unigram model: expecting 1 word input
            if model.n == 1 and len(words) != 1:
                print(f"Error: --word must contain 1 word for this unigram model")

            # Bigram model: expecting 2 words input
            elif model.n == 2 and len(words) != 2:
                print(f"Error: --word must contain 2 words for this bigram model")
                return
            ngram = tuple(words)
            deterministic = args.d if 'd' in args else False
            for _ in range(args.nwords):
                next_word = model.predict_next_word(ngram, deterministic=deterministic)
                print(next_word, end=' ')
                if model.n > 1:  # Bigram model
                    ngram = (*ngram[1:], next_word)
                else:  # Unigram model
                    ngram = (next_word, )

            print()
        except:
            print("There was an error opening the model.")

    elif args.activity == "train_bpe":
        if not args.data:
            print("Error: --data is required for train_bpe")
            return
        with open(args.data, 'r', encoding='utf-8') as file:
            data = file.read()
        model = BPE()
        model.train(data)
        if args.save:
            with open(args.save, 'wb') as file:
                pickle.dump(model, file)
        print("BPE model trained and saved.")

    elif args.activity == "tokenize":
        if not args.load or not args.text:
            print("Error: --load and --text are required for tokenize")
            return
        print(f"Loading model from: {args.load}")
        print(f"Text to tokenize: {args.text}")
        with open(args.load, 'rb') as file:
            model = pickle.load(file)
        tokens, token_ids = model.tokenize(args.text)  # This line should now be safe
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)






if __name__ == "__main__":
    main()



