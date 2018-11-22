import array
import sys

import jenks


def main():

    # Read states and build states array
    inputFilename = str(sys.argv[1])
    iFile = open(inputFilename,"r+")

    lines = iFile.readlines()
    lines.pop(0)

    data = list()
    for i in range(1,10000):
    	words = lines[i].split(',')
    	data.append(float(words[7]))

    iFile.close()
    
    # clusterize
    print(jenks.getJenksBreaks(data,5))

    # print states out 


if __name__ == "__main__":
    main()
