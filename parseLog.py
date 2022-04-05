import numpy as np, scipy.stats as st
import sys

# Open input file
inputFileName = sys.argv[1]
inputFile = open(inputFileName, "r")

# Create samples array
results = {
   "dataBER": np.array([]),
   "decodedDataBER": np.array([]),
   "dataLoss": np.array([]),
   "dataCRCFailure": np.array([]),
   "controlBER": np.array([]),
   "decodedControlBER": np.array([]),
   "controlLoss": np.array([]),
   "controlCRCFailure": np.array([]),
   "reportedCollision": np.array([]),
   "reportedLowSNR": np.array([]),
}

# Ignore the first line
inputFile.readline()

# Read the rest of the file
for line in inputFile:

    # If it is the header, ignore it.
    if line[0] == '#':
        continue 

    # No, parse the line.
    # Split line into fiels
    fields = line.split()

    # Append the value of each field to the corresponding array of results
    results["dataBER"] = np.append(results["dataBER"], float(fields[0]))
    results["decodedDataBER"] = np.append(results["decodedDataBER"], float(fields[1]))
    results["dataLoss"] = np.append(results["dataLoss"], float(fields[2]))
    results["dataCRCFailure"] = np.append(results["dataCRCFailure"], float(fields[3]))
    results["controlBER"] = np.append(results["controlBER"], float(fields[4]))
    results["decodedControlBER"] = np.append(results["decodedControlBER"], float(fields[5]))
    results["controlLoss"] = np.append(results["controlLoss"], float(fields[6]))
    results["controlCRCFailure"] = np.append(results["controlCRCFailure"], float(fields[7]))

    # Process statistics about the collision detection 
    reportedCollision = 0
    reportedLowSNR = 0
    if int(fields[2]) == 1:
        # Data was lost .
        if int(fields[6]) == 1:
            # So was control. We report a collision.
            reportedCollision = 1
        else:
            # Control was received. Report low SNR.
            reportedLowSNR = 1

    results["reportedCollision"] = np.append(results["reportedCollision"], reportedCollision)
    results["reportedLowSNR"] = np.append(results["reportedLowSNR"], reportedLowSNR)

inputFile.close()

#print results
# Compute statistics for each column
for col in results:
    m = np.mean(results[col])
    ci = st.t.interval(0.95, len(results[col])-1, loc=m, scale=st.sem(results[col]))
    print ("%s: %.6f +- %.6f" % (col, m, m - ci[0]))

