import csv
import os
import keyboard
import cv2
import sys
csv_path = "labels.csv"
pictures_path = "cropped"
csv_map = dict()
if(os.path.exists(csv_path)):
    with open(csv_path) as file:
        my_csv = csv.reader(file)
        for line in my_csv:
            csv_map[line[0]] = line[1]

    with open(csv_path,"a") as out_file:
        
        for subdir, dirs, files in os.walk(pictures_path):
            for file in files:
                if(file not in csv_map.keys()):
                    answer = ""
                    img = cv2.imread(os.path.join(subdir,file)) 
                    cv2.namedWindow("Does this have a seam? 0 for no, 1 for yes", cv2.WINDOW_NORMAL)
                    cv2.imshow("Does this have a seam? 0 for no, 1 for yes", img)
                    while True:

                        cv2.waitKey(0)
                        answer = keyboard.read_key()
                        print(answer)
                        if(answer == "1" or answer == "0"):
                            break;
                        elif(answer== "esc"):
                            sys.exit("Quitting")
                    out_file.write(file + "," + answer + "\n")
                
else:               
    print("no csv")