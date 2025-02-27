import sys, re, collections

# Checkpoint 1.1
def wordTokenizer(sent):

    #input: a single sentence as a string.

    #output: a list of each “word” in the text

    # must use regular expressions

    tokens = [] 

    # Check to retain abbreviations of capital letters e.g U.S.A.
    abbrevs = re.findall(r'([A-Z]\.)+(([A-Z]\.)+)?', sent)
    for i, abbr in enumerate(abbrevs):
        sent = sent.replace(abbr, f"ABBR{i}",1)

    # Check to retain periods surrounded by integers e.g 6.0    
    nums = re.findall(r'(\d+\.\d+)', sent)
    for i, num in enumerate(nums):
        sent = sent.replace(num, f"NUM{i}",1)

    # Check to retain contractions e.g. don't, can't, won't, etc.
    contractions = re.findall(r"(\w+'\w+)", sent)
    for i, con in enumerate(contractions):
        sent = sent.replace(con, f"CON{i}",1)

    # Check to retain hashtags and @mentions e.g. #bestie, @bestie
    hashtags = re.findall(r'(#\w+)', sent)
    for i, tag in enumerate(hashtags):
        sent = sent.replace(tag, f"TAG{i}",1)

    mentions = re.findall(r'(@\w+)', sent)
    for i, ment in enumerate(mentions):
        sent = sent.replace(ment, f"MENT{i}",1)

    # Separate all punctuation from words
    sent = re.sub(r'([^\w\s])', r' \1 ', sent)

    # Replace all spaces with a single space
    sent = re.sub(r'\s+', ' ', sent).strip()

    # Return all placeholders to their original form
    for i, abbr in enumerate(abbrevs):
        sent = sent.replace(f"ABBR{i}", abbr)
    for i, num in enumerate(nums):
        sent = sent.replace(f"NUM{i}", num)
    for i, con in enumerate(contractions):
        sent = sent.replace(f"CON{i}", con)
    for i, tag in enumerate(hashtags):
        sent = sent.replace(f"TAG{i}", tag)
    for i, ment in enumerate(mentions):
        sent = sent.replace(f"MENT{i}", ment)

    return sent.split()

# Checkpoint 1.2
def spacelessBPELearn(docs, max_vocabulary=1000):
    
    #input: docs, a list of strings to be used as the corpus for learning the BPE vocabulary

    #output: final_vocabulary, a set of all members of the learned vocabulary

    # Start with vocabulary of all ascii letters as words
    vocab = set() 
    vocab.update(chr(i) for i in range(97, 123))  # a-z
    vocab.update(chr(i) for i in range(65, 91))  # A-Z
    
    # Convert non-ascii to ?
    words = []
    for d in docs:
        d = ''.join(c if c.isascii() else '?' for c in d)
        newWords = d.split()
        words.extend(newWords)
    
    # Convert word to chars list
    tokens = [[c if c.isascii() else '?' for c in word] for word in words]

    iteration=0

    targetIterations=[0,1,10,100,500]
    
    # Merge tokens in BPE algorithm
    while len(vocab) < max_vocabulary:
        # Most frequent bigram in set of strings
        pairs = collections.defaultdict(int)
        for word in tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] += 1
        
        if not pairs:
            break
        
        # Get the most frequent bigram
        mostFrequentBigram = max(pairs.items(), key=lambda x: x[1])[0]
        
        # Create a new token by combining the pair
        newToken = ''.join(mostFrequentBigram)
        vocab.add(newToken)
        
        # Replace each occurrence of the pair with new token
        newWordTokens = []
        for word in tokens:
            i = 0
            newWord = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == mostFrequentBigram:
                    newWord.append(newToken)
                    i += 2
                else:
                    newWord.append(word[i])
                    i += 1
            newWordTokens.append(newWord)
        
        tokens = newWordTokens

        if iteration in targetIterations:
            print(f"Iteration {iteration}:")
            topPairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (pair, count) in enumerate(topPairs):
                print(f"  {i+1}. Pair: {pair} -> {''.join(pair)} (count: {count})")
        iteration += 1
    
    return vocab

def spacelessBPETokenize(text: str, vocab):

    #input: text, a single string to be word tokenized.

    #       vocab, a set of valid vocabulary words

    #output: words, a list of strings of all word tokens, in order, from the string

    # Replace non-ASCII characters with '?'
    text = ''.join(c if c.isascii() else '?' for c in text)
    
    rawWords = text.split()
    res = []
    
    # Process
    for word in rawWords:
        chars = list(word)
        
        while True:
            pairs = [(chars[i], chars[i+1]) for i in range(len(chars) - 1)]

            validMerges = [(i, ''.join(pairs[i])) for i in range(len(pairs)) if ''.join(pairs[i]) in vocab]
            
            if not validMerges:
                break
            
            pos, merged = validMerges[0]
            newChars = chars[:pos] + [merged] + chars[pos+2:]
            chars = newChars
        
        res.extend(chars)
    
    return res

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python a1_p1_mahmood_113472709.py <train_file>")
        sys.exit(1)
    
    # Grab input file and read in the data
    file = sys.argv[1]
    docs = []
    with open(file, 'r', encoding='utf-8') as f:
        docs = [line.strip() for line in f.readlines()]
    
    # Checkpoint 1.1 
    # Print first 5 and last doc
    print("Checkpoint 1.1:")
    for i in range(0,5):
        print(wordTokenizer(docs[i]))
    print(wordTokenizer(docs[-1]))


    # Checkpoint 1.2 
    print("\nCheckpoint 1.2:")
    # Print top 5 most frequent pairs at iterations 0,1, 10, 100 and 500
    vocabulary = spacelessBPELearn(docs)
    # Print final vocabulary
    print(f"Final vocabulary: {list(vocabulary)}...")
    # Print first 5 and last doc
    for i in range(5):
        print(f"{spacelessBPETokenize(docs[i], vocabulary)}")
    
    print(f"{spacelessBPETokenize(docs[-1], vocabulary)}")

    pass