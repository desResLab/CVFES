#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    CVConfig is the configuration structure of the CVFES project.
"""

from configobj import ConfigObj

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


class Config:

    def __init__(self, config):
        try:
            self.name = config['name']
            self.file_path = config['file_path']
        except KeyError as ex:
            print('Key {} does not exit!'.format(ex))


class FaceConfig(Config):

    def __init__(self, faceSection):
        Config.__init__(self, faceSection)


class ConditionConfig:

    def __init__(self):
        pass

    def getProp(self, config, key):
        try:
            subConfig = config[key]
            if 'uniform_value' in subConfig:
                prop = subConfig.as_float('uniform_value')
            elif 'file_name' in subConfig:
                prop = subConfig['file_name']
            else:
                print('Failed to get initial conditions configuration for {}\'s {}.'.format(config.parent.name, key))
        except KeyError as ex:
            print('Key {} does not exist!'.format(ex))

        return prop

class InitialConditionsConfig(ConditionConfig):

    def __init__(self, iniCndSection):

        ConditionConfig.__init__(self)

        self.acceleration = self.getProp(iniCndSection, 'acceleration')
        self.velocity = self.getProp(iniCndSection, 'velocity')
        self.pressure = self.getProp(iniCndSection, 'pressure')
        self.displacement = self.getProp(iniCndSection, 'displacement')

class BoundaryConditionsConfig(ConditionConfig):

    def __init__(self, bdyCndSection):

        ConditionConfig.__init__(self)

        self.traction = self.getProp(bdyCndSection, 'traction')


class MeshConfig(Config):

    def __init__(self, meshSection):
        Config.__init__(self, meshSection)
        self.domainId = meshSection.as_int('domain_id')
        self.thickness = meshSection.as_float('thickness')
        self.density = meshSection.as_float('density')
        self.E = meshSection.as_float('Youngs Modulus')
        self.v = meshSection.as_float('Poissons Ratio')

        self.faces = []
        if 'faces' in meshSection:
            facesConfig = meshSection['faces']
            for section in facesConfig.sections:
                if section.startswith('face'):
                    self.faces.append(FaceConfig(facesConfig[section]))

        self.initialConditions = InitialConditionsConfig(meshSection['initial conditions'])
        self.boundaryConditions = BoundaryConditionsConfig(meshSection['boundary conditions'])

        # DEBUG:
        # print 'Mesh {} contains {} faces in domain {}\n'.format(self.name, len(self.faces), self.domainId)

        # Config the result filename.
        self.stressFilename = meshSection['stress_filename']
        self.saveResNum = meshSection['save_result_num']

class SolverConfig():

    def __init__(self, solverSection):
        self.method = solverSection['method']
        self.time = solverSection.as_float('time')
        self.dt = solverSection.as_float('dt')
        self.endtime = solverSection.as_float('endtime')
        self.tolerance = solverSection.as_float('tolerance')

class CVConfig:

    def __init__(self, config):
        """ Walk through the ConfigObj object to
        collect configuration informations. """

        # Constant parameters.
        # Meshes.
        self.meshes = []
        try:
            meshesConfig = config['meshes']
            for section in meshesConfig.sections:
                if section.startswith('mesh'):
                    self.meshes.append(MeshConfig(meshesConfig[section]))
        except KeyError as ex:
            print('Key {} does not exits!'.format(ex))

        # Solver parameters.
        self.solver = SolverConfig(config['solver'])

