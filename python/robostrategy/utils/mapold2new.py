import numpy as np
import kaiju.robotGrid


def mapold2new():
    
    rgold = kaiju.robotGrid.RobotGridFilledHex()
    rgnew = kaiju.robotGrid.RobotGridAPO()

    xposnew = np.array([rgnew.robotDict[r].xPos 
                        for r in rgnew.robotDict])
    yposnew = np.array([rgnew.robotDict[r].yPos 
                        for r in rgnew.robotDict])
    ridnew = np.array([rgnew.robotDict[r].id 
                       for r in rgnew.robotDict])

    old2new = dict()
    new2old = dict()
    for ridold in rgold.robotDict:
        rold = rgold.robotDict[ridold]
        inew = np.where((np.abs(rold.xPos - xposnew) < 1.0) &
                        (np.abs(rold.yPos - yposnew) < 1.0))[0][0]
        old2new[ridold] = ridnew[inew]
        new2old[ridnew[inew]] = ridold
        
    return(old2new, new2old)
