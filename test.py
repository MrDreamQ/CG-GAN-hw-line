transcr = ' Become a '
pair_idx = [(1,6)]
for h, t in pair_idx:
    word_transcr = ' ' * len(transcr[:h]) + transcr[h:t+1] + ' ' * len(transcr[t + 1:])
    print(word_transcr)