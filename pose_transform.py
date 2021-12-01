# -*-coding:utf-8-*-
fo = open("./09.txt", "r+")
new = open("./09_new.txt", "w")
for line in fo.readlines():
    # print(line)
    # print(line.split(" "))
    newline = line.split(" ")
    print(newline)
    # print(type(newline[0][0]))
    for i in range(12):
        print(i)
        print(newline[i])
        if i != 0:
            new.write(" ")
        if i == 11 and (newline[i][0] != "-"):
            new.write("-")
            new.write(newline[i])
        elif i == 11 and (newline[i][0] == "-"):
            new.write(newline[i][1:])
            print(newline[i][1:])
        else:
            new.write(newline[i])
            print(newline[i])

    # new.write("\n")

new.close()
fo.close()