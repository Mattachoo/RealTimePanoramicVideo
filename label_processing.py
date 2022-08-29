import csv
label_map = dict()

with open(r"C:\Users\mattp\Documents\thesis_Stuff\dist\label\labels.csv") as my_file:
    #csv_file = csv.reader(my_file)
    csv_file =my_file.readlines()
    for row_ in csv_file:
        row = row_.split(",")
        if "Copy" not in row[0]:
            label_map[row[0]] = row[1]
with open("label_filtered.csv", 'a') as my_file2:

    for key in label_map.keys():
        print(key)
        my_file2.write(key + "," + label_map[key])