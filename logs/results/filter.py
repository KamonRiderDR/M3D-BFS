with open("HCP_thresh.txt", "r") as file:
    start = 41
    length = 12
    for line in file:
        # print(line)
        if line is not None:
            print(line[start:start+length])
file.close()