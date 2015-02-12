#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import python libs
from multiprocessing import Process, Event, Queue
from pprint import pprint as pp
from sys import stdout
from array import array
import multiprocessing
import argparse
import numpy as np
import matplotlib.image as image

# defining constants
BE_VERBOSE = False
ITERATIONS = 1024
NEWTON_MAX_ITERATIONS = 50
PROCESS_COUNT = (multiprocessing.cpu_count() - 1)
OUTPUT_FILENAME = 'NewtonFractal'
LOOP_AND_ZOOM = False
ZOOM_FACTOR = 0.95



class NewtonFractalProcessController():
    def __init__(self):

        self.complexMatrixBorder = 1
        self.loopAndZoomIteration = 0

        # It is kind of dirty to just loop the whole calculation process
        # and not reusing the existing worker processes. To do so, implement
        # a new way of passing the complex matrix to the workers.
        # But ain't nobody got time for that...
        while LOOP_AND_ZOOM or self.loopAndZoomIteration < 1:
            # calculation initializations
            self.createComplexMatrix()
            self.colorMatrix = np.zeros((ITERATIONS, ITERATIONS, 3))
            self.currentCalculatedPixelRow = 0

            # system initializations
            self.createWorkerProcesses()
            self.lastAsignedPixelRowIndex = [0 for i in range(0, PROCESS_COUNT)]

            # start calculation
            if BE_VERBOSE: print('Start calculation of ' + str(ITERATIONS) +
                ' pixels in square with ' + str(PROCESS_COUNT) + ' processes.')
            self.controlCalculation()

            # create and save the image from the color matrix
            self.createImage()

            self.loopAndZoomIteration = self.loopAndZoomIteration + 1

    def controlCalculation(self):
        currentProzessIndex = 0
        while self.currentCalculatedPixelRow < ITERATIONS:
            if not self.workerCalculationEvents[currentProzessIndex].is_set():

                # get the calculated pixel row and put it into the color matrix
                if self.usedWorkerWithIndex[currentProzessIndex]:
                    self.mergePixelRowToColorMatrixFromWorkerWithIndex(currentProzessIndex)

                # set new pixel line to calculate to process
                self.workerCommunicationQueues[currentProzessIndex].put(self.currentCalculatedPixelRow)
                self.lastAsignedPixelRowIndex[currentProzessIndex] = self.currentCalculatedPixelRow
                self.currentCalculatedPixelRow = self.currentCalculatedPixelRow + 1
                if BE_VERBOSE:
                    stdout.write("\r" + 'calculation progress: ' +
                        str(int((self.currentCalculatedPixelRow / ITERATIONS) * 100)) + '%')
                    stdout.flush()

                # start new calculation
                self.workerCalculationEvents[currentProzessIndex].set()
                self.usedWorkerWithIndex[currentProzessIndex] = True

            # adjust process index for next iteration
            if currentProzessIndex < (PROCESS_COUNT - 1):
                currentProzessIndex = currentProzessIndex + 1
            else:
                currentProzessIndex = 0

        # get the pixel rows of the last iterations
        for i in range(0, PROCESS_COUNT):
            self.mergePixelRowToColorMatrixFromWorkerWithIndex(i)

    def mergePixelRowToColorMatrixFromWorkerWithIndex(self, workerIndex):
        pixelRowIndex = self.lastAsignedPixelRowIndex[workerIndex]
        calculatedPixelRow = self.workerCommunicationQueues[workerIndex].get()

        # integrate calculated pixel row into the color matrix
        self.colorMatrix[pixelRowIndex] = calculatedPixelRow

    def createComplexMatrix(self):
        if self.loopAndZoomIteration > 0:
            self.complexMatrixBorder = self.complexMatrixBorder * ZOOM_FACTOR

        self.complexMatrix = np.zeros((ITERATIONS, ITERATIONS), dtype='complex64')
        baseVector = np.linspace(-self.complexMatrixBorder, self.complexMatrixBorder, ITERATIONS)
        for oneRow in range(0, ITERATIONS):
            self.complexMatrix[oneRow] = baseVector + baseVector[(ITERATIONS - 1) - oneRow] * 1j

    def createWorkerProcesses(self):
        self.workerCommunicationQueues = []
        self.workerCalculationEvents = []

        self.workerProcesses = []
        self.usedWorkerWithIndex = []

        for i in range(0, PROCESS_COUNT):
            processCommunicationQueue = Queue()
            startEvent = Event()
            finishEvent = Event()
            finishEvent.set()
            process = NewtonFractalWorkerProcess(processCommunicationQueue,
                                                startEvent, finishEvent,
                                                self.complexMatrix.T)
            process.start()

            self.workerCommunicationQueues.append(processCommunicationQueue)
            self.workerCalculationEvents.append(startEvent)

            self.workerProcesses.append(process)
            self.usedWorkerWithIndex.append(False)

    def createImage(self):
         # normalize colors
        normalizationDivisor = np.max(self.colorMatrix)
        if BE_VERBOSE: print('\nnormalizing lightness with divisor: ' +
            str(normalizationDivisor))
        self.colorMatrix = np.divide(self.colorMatrix, normalizationDivisor)

        # save image
        if LOOP_AND_ZOOM:
            currentFileName = OUTPUT_FILENAME + str(self.loopAndZoomIteration) + '.png'
        else:
            currentFileName = OUTPUT_FILENAME + '.png'
        image.imsave(currentFileName, self.colorMatrix)
        if BE_VERBOSE: print('saved image as "' + currentFileName + '"')


class NewtonFractalWorkerProcess(Process):
    def __init__(   self, processCommunicationQueue,
                    processCalculatingFlag, finishCalculationEvent,
                    complexNumberMatrix):

        # system initializations
        self.processCommunicationQueue = processCommunicationQueue
        self.processCalculatingFlag = processCalculatingFlag
        self.finishCalculationEvent = finishCalculationEvent

        # calculation initializations
        self.complexNumberMatrix = complexNumberMatrix
        self.convergenceCriteria = 10 * np.finfo(np.float64).eps
        self.blackPixel = np.zeros((3))
        self.colorMatrixPartial = np.zeros((ITERATIONS, 3))
        self.pixelLineToCalculate = 0

        # Avoid using np.poly1d(), they're a performance killer!
        # Instead, write down the roots statical and define the function.
        # Use 'np.roots(np.poly1d([1, 0, 0, 0, 0, 0, 0, 0, -1]))'
        # to calculate the roots array.
        # self.roots = np.array([-0.5+0.8660254j, -0.5-0.8660254j,  1.0+0.j])
        self.roots = np.array([ -1.00000000e+00+0.j,   8.32667268e-17+1.j,   8.32667268e-17-1.j, 1.00000000e+00+0.j])

        # instantiate as process
        Process.__init__(self)

    def function(self, z):
        # return z**3 - 1
        return z**4 - 1

    def dfunction(self, z):
        # return 3 * z**2
        return 4 * z**3

    def run(self):
        # after each calculation of one row, wait until the controller process
        # put a new pixel row index into the queue and set the flag to start
        while self.processCalculatingFlag.wait(1):

            # get new line to calculate from the comm. queue
            self.pixelLineToCalculate = self.processCommunicationQueue.get()

            # do the actual calculation for one pixel row
            for j in range(0, ITERATIONS):
                (root, iterations) = self.findRootNewton(self.complexNumberMatrix[self.pixelLineToCalculate][j])

                for r in range(len(self.roots)):
                    if np.around(root - self.roots[r]) == 0:
                        if r % 6 < 3:
                            self.colorMatrixPartial[j] = self.blackPixel
                            self.colorMatrixPartial[j][r % 6] = iterations
                        elif r % 6 == 3:
                            self.colorMatrixPartial[j] = self.blackPixel
                            self.colorMatrixPartial[j][0] = iterations
                            self.colorMatrixPartial[j][1] = iterations
                        elif r % 6 == 4:
                            self.colorMatrixPartial[j] = self.blackPixel
                            self.colorMatrixPartial[j][0] = iterations
                            self.colorMatrixPartial[j][2] = iterations
                        elif r % 6 == 5:
                            self.colorMatrixPartial[j] = self.blackPixel
                            self.colorMatrixPartial[j][1] = iterations
                            self.colorMatrixPartial[j][2] = iterations
                        break

            # set the calculated row to the queue to make it accessible for the controller process
            self.processCommunicationQueue.put(self.colorMatrixPartial)

            # tell controller process (passive) that the calculation is done
            self.processCalculatingFlag.clear()

    def findRootNewton(self, root):
        currentIterationCount = 0

        while(np.abs(self.function(root)) > self.convergenceCriteria):
            root = root - (self.function(root) / self.dfunction(root))
            currentIterationCount += 1
            if currentIterationCount >= NEWTON_MAX_ITERATIONS: break

        return root, currentIterationCount



# ************************************************
# non object orientated entry code goes down here:
# ************************************************
# check if this code is run as a module or was included into another project
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Multiprocess Newton fractal calculation")
    parser.add_argument("-v", "--verbose", action = "store_true", dest = "verbose", help = "enables verbose mode")
    parser.add_argument("-l", "--loop", action = "store_true", dest = "loop", help = "starts infinite calculation loop for many frames")
    parser.add_argument("-i", "--iterations", type=int, help = "specifies the number of iterations")
    parser.add_argument("-p", "--processcount", type=int, help = "specifies the number of processes")
    args = parser.parse_args()

    if args.verbose:
        BE_VERBOSE = True

    if args.loop:
        LOOP_AND_ZOOM = True

    if args.iterations:
        ITERATIONS = args.iterations

    if args.processcount:
        PROCESS_COUNT = args.processcount

    # ensure that at least one woker process is spawned
    if PROCESS_COUNT < 1:
        PROCESS_COUNT = 1

    processController = NewtonFractalProcessController()
