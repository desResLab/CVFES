import numpy as np
import os.path

from mesh import *

# last color is reserved
def MeshColoring(name, mesh, nColors=23, maxNeighbors=50):

    nNodes = mesh.lclNNodes
    nElms = mesh.nElements
    elmNodeIds = mesh.lclElmNodeIds

    # Try to load the coloring result first.
    if os.path.exists('{}.npz'.format(name)):
        colorData = np.load('{}.npz'.format(name), allow_pickle=True)
        mesh.elmColors = colorData['elmColors']
        mesh.colorGroups = colorData['colorGroups']
        return 0


    # If failed, generate the color group and save in file.
    # 1. Find the neighbors.
    nodeElmNeighbors = np.empty((nNodes, maxNeighbors), dtype=int)
    nodeElmNeighborsCount = np.zeros(nNodes, dtype=int)
    for iElm in range(nElms):
        for iNode in elmNodeIds[iElm]:
            nodeElmNeighbors[iNode,nodeElmNeighborsCount[iNode]] = iElm
            nodeElmNeighborsCount[iNode] += 1

    neighbors = [[] for _ in range(nElms)]
    for iElm in range(nElms):
        for iNode in elmNodeIds[iElm]:
            neighbors[iElm].extend(nodeElmNeighbors[iNode,:nodeElmNeighborsCount[iNode]])
    # Clear the redundent neighbors.
    neighbors = np.array(neighbors)
    for i in range(nElms):
        neighbors[i] = np.unique(neighbors[i])

    # 2. Label the colors.
    elmColors = np.empty(nElms)
    neighborColors = [[] for _ in range(nElms)]
    for i in range(nElms):
        for iColor in range(nColors+1):
            if iColor not in neighborColors[i]:
                break

        if iColor == nColors:
            print('Colors are used up! {} colors are not enough.'.format(nColors))
            return -1

        elmColors[i] = iColor
        for iNeighbor in neighbors[i]:
            neighborColors[iNeighbor].append(iColor)
    # Put each group together.
    nColors = int(elmColors.max()) + 1
    colorGroups = np.array([np.where(elmColors == iColor)[0] for iColor in range(nColors)])

    # 3. Save the result into files.
    mesh.elmColors = elmColors
    mesh.colorGroups = colorGroups
    np.savez(name, elmColors=elmColors, colorGroups=colorGroups)

    return 0
