import os
import csv

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    print(listOfFile)
    allFiles = list()
    allFiles.append(["class","char"])
    i=0
    # Iterate over all the entries
    for entry in listOfFile:
        d=entry.split("_")
        print(len(d))
        print(d[len(d)-1])
        allFiles.append([i,d[len(d)-1]])
        i=i+1
                
    return allFiles




dirName = 'Dataset';
 
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
print(listOfFiles)


with open('sample.csv', 'w',newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(listOfFiles)
