# -*-coding:utf-8-*-
# test_files_00.txt
new = open("./splits/odom/test_files_11.txt", "w")
n = 1091
for i in range(n):
    new.write('11 ')
    new.write(str(i).zfill(8))
    new.write(' l')
    new.write('\n')

print("done")