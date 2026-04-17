import csv

def importCSV(filename, linesTobeIgnored = 0, columnsToBeIgnored = 0):
    result = []
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        for i in range(linesTobeIgnored):
            next(reader)
        for row in reader:
            result.append(row[columnsToBeIgnored:])

    return result

def importTxt(filename, linesTobeIgnored = 0, columnsToBeIgnored = 0, separator = " "): 
    result = []
    with open(filename, "r", encoding='utf-8') as file:
        for _ in range(linesTobeIgnored):
            file.readline()
        for line in file:
            result.append(line.strip().split(separator)[columnsToBeIgnored:])
    return result