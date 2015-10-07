#!/usr/bin/python
#Way of Use

import os
import sys
import argparse
import re
import string
import logging

#from nyural import net
from nyural import solver
from nyural.protoc import nyural_pb2
from google.protobuf import text_format

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Deep Neural Network with Hessian Free Optimization')
    parser.add_argument(
        'action',
        help='Input for training (train) or testing(test)')
    parser.add_argument(
        '-solver',
        help='File containing the solver parameters')
    parser.add_argument(
        '-log_dir',
        help='Optional, Directory to save the log info')
    args = parser.parse_args()
    return args

def ReadSolverFromTextFile(textfile,logger):
    logger.info("Reading Solver From:" + textfile)
    param = nyural_pb2.SolverParameter()
    f = open(textfile,"r") #rb
    text_format.Merge(str(f.read()), param)
    f.close()
    return param

def ReadNetFromTextFile(textfile):
    param = nyural_pb2.NetParameter()
    f = open(textfile,"rb") #rb
    text_format.Merge(str(f.read()), param)
    f.close()
    return param

def main():
    args = parse_args()
    print("Deep Neural Network Training using Hessian Free")
    if args.action == "train":
        #Setting Log Configs
        logger = logging.getLogger('myapp')
		
        hdlr = logging.FileHandler('myapp.log')
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        logger.setLevel(logging.DEBUG)

        logger.info("Begining the training")

        #Init Vars
        SolverParam = ReadSolverFromTextFile(args.solver,logger)
        NetParam = ReadNetFromTextFile(SolverParam.net)
        Solver = solver.solver(SolverParam,logger)
        Solver.InitNet(NetParam)
        Solver.solve()
        #if SolverParam.gpu > 0:

    if args.action == "test":
        logger = logging.getLogger('myapp')
if __name__ == '__main__':
    main()


