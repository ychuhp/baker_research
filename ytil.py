def file__list(writeList,filename):
    with open(filename, 'w') as f:
        for item in writeList:
            f.write("%s\n" % item)

def list__file(filename):
    returnList = []
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            line = line.rstrip('\n')
            returnList.append(line)
    return returnList
