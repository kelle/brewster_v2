import numpy as np

from sizes import *

def read_cia(filename, inwavenum, nwave):

    outcia = np.zeros((4, nciatemps, maxwave), dtype=np.float32)
    ciatemps = np.zeros(nciatemps)
    oldcia = np.zeros((4, nciatemps, ncwave))
    ciawaven = np.zeros(ncwave, dtype=np.float64)
    wdiff = np.zeros(ncwave, dtype=np.float64)

    with open(filename, 'r') as f:
        for i in range(3):
            f.readline()
        idum1, idum2 = map(int, f.readline().split())

        if idum1 != ncwave or idum2 != nciatemps:
            print("Problem with low-res CIA table : ", filename.strip())
            exit()

        ciaarray = np.zeros((4, nciatemps, nwave))

        for iciatemp in range(1, nciatemps + 1):
            ciatemps[iciatemp - 1] = (f.readline().strip())
            for icwaven in range(1, ncwave + 1):
                line = f.readline().split()
                ciawaven[icwaven - 1] = float(line[0])
                oldcia[0, iciatemp - 1, icwaven - 1] = float(line[1])
                oldcia[1, iciatemp - 1, icwaven - 1] = float(line[2])
                oldcia[2, iciatemp - 1, icwaven - 1] = float(line[3])
                oldcia[3, iciatemp - 1, icwaven - 1] = float(line[4])

        for iwave in range(nwave):
            wdiff = np.abs(ciawaven - inwavenum[iwave])
            oldw1 = np.argmin(wdiff)

            if (ciawaven[oldw1] < inwavenum[iwave]):
                oldw2 = oldw1 + 1
            else:
                oldw2 = oldw1
                oldw1 = oldw2 - 1

            intfact = (np.log10(inwavenum[iwave]) - np.log10(ciawaven[oldw1])) / (
                        np.log10(ciawaven[oldw2]) - np.log10(ciawaven[oldw1]))

            ciaarray[0, :, iwave] = ((oldcia[0, :, oldw2] - oldcia[0, :, oldw1]) * intfact) + oldcia[0, :, oldw1]
            ciaarray[1, :, iwave] = ((oldcia[1, :, oldw2] - oldcia[1, :, oldw1]) * intfact) + oldcia[1, :, oldw1]
            ciaarray[2, :, iwave] = ((oldcia[2, :, oldw2] - oldcia[2, :, oldw1]) * intfact) + oldcia[2, :, oldw1]
            ciaarray[3, :, iwave] = ((oldcia[3, :, oldw2] - oldcia[3, :, oldw1]) * intfact) + oldcia[3, :, oldw1]

            for i in range(4):
                for iciatemp in range(nciatemps):
                    outcia[i, iciatemp, iwave] = ciaarray[i, iciatemp, iwave]

    return outcia, ciatemps
