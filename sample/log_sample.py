import os
import sys
import logging 

logger = logging.getLogger("new")

formatter = logging.Formatter('%(message)s')
fileHandler = logging.FileHandler("new.log")
fileHandler.setLevel(logging.DEBUG)

fileHandler.setFormatter(formatter)

logger.addHandler(fileHandler)


if __name__=="__main__":

    logger.error("Created a different sucessfully")

    
