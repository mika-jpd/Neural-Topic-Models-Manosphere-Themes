with open("subreddits/socialanxiety2.txt", "r") as f:
    words = set()
    avg_len = 0
    cnt = 0

    for line in f.readlines():
        current_words = line.split(" ")
        avg_len += len(current_words)
        cnt += 1

        for word in current_words:
            words.add(word.lower())
    
    
    print(len(words), avg_len / cnt)