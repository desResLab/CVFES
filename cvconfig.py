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


class MeshConfig(Config):

    def __init__(self, meshSection):
        Config.__init__(self, meshSection)
        self.domainId = meshSection.as_int('domain_id')

        self.faces = []
        if 'faces' in meshSection:
            facesConfig = meshSection['faces']
            for section in facesConfig.sections:
                if section.startswith('face'):
                    self.faces.append(FaceConfig(facesConfig[section]))

        # DEBUG:
        # print 'Mesh {} contains {} faces in domain {}\n'.format(self.name, len(self.faces), self.domainId)

class ConditionConfig:

    def __init__(self):
        pass

    def getProp(self, config, key):
        try:
            subConfig = config[key]
            if 'uniform_value' in subConfig:
                prop = subConfig.as_float('uniform_value')
            elif 'uniform_func' in subConfig:
                prop = subConfig['uniform_func']
            elif 'file_name' in subConfig:
                prop = subConfig['file_name']
            else:
                print('Failed to get initial conditions configuration for {}\'s {}.'.format(config.parent.name, key))
        except KeyError as ex:
            print('Key {} does not exist!'.format(ex))

        return prop

class InitialConditionsConfig(ConditionConfig):

    def __init__(self, iniCndSection, name):

        ConditionConfig.__init__(self)

        if name == 'fluid':
            self.acceleration = self.getProp(iniCndSection, 'acceleration')
            self.velocity = self.getProp(iniCndSection, 'velocity')
            self.pressure = self.getProp(iniCndSection, 'pressure')
        else:
            self.velocity = self.getProp(iniCndSection, 'velocity')
            self.displacement = self.getProp(iniCndSection, 'displacement')

class BoundaryConditionsConfig(ConditionConfig):

    def __init__(self, bdyCndSection, name):

        ConditionConfig.__init__(self)

        if name == 'fluid':
            self.inletVelocity = self.getProp(bdyCndSection, 'inlet_velocity')
            self.outletH = self.getProp(bdyCndSection, 'outlet_h')
            self.parabolicInlet = False
            if 'parabolic_inlet' in bdyCndSection:
                self.parabolicInlet = bdyCndSection.as_bool('parabolic_inlet')
        else:
            self.bdyDisplacement = self.getProp(bdyCndSection, 'displacement')

class EquationConfig():

    def __init__(self, equationSection):

        if equationSection['name'] == 'fluid':
            try:
                self.dviscosity = equationSection.as_float('dynamic_viscosity')
                self.density = equationSection.as_float('density')
                self.f = equationSection.as_float('external_force')
            except KeyError as ex:
                print('Key {} does not exits!'.format(ex))
        else:
            try:
                self.thickness = equationSection.as_float('thickness') # TODO: to be reconsidered!!!!!!!!!!!!!!!!!!!!
                self.density = equationSection.as_float('density')
                self.E = equationSection.as_float('Youngs Modulus')
                self.v = equationSection.as_float('Poissons Ratio')
                self.damp = equationSection.as_float('Damp')
            except KeyError as ex:
                print('Key {} does not exits!'.format(ex))

            self.thicknessFilename = None
            self.sigmaThickness = 0.0
            self.rhoThickness = 3.7
            if 'thickness Filename' in equationSection:
                self.thicknessFilename = equationSection['thickness Filename']
                self.sigmaThickness = equationSection.as_float('thickness sigma')
                self.rhoThickness = equationSection.as_float('thickness rho')

            self.YoungsModulusFilename = None
            self.sigmaE = 0.0
            self.rhoE = 3.7
            if 'Youngs Modulus Filename' in equationSection:
                self.YoungsModulusFilename = equationSection['Youngs Modulus Filename']
                self.sigmaE = equationSection.as_float('Youngs Modulus sigma')
                self.rhoE = equationSection.as_float('Youngs Modulus rho')

        self.initialConditions = InitialConditionsConfig(equationSection['initial conditions'], equationSection['name'])
        self.boundaryConditions = BoundaryConditionsConfig(equationSection['boundary conditions'], equationSection['name'])


class SolverConfig():

    def __init__(self, solverSection):
        self.method = solverSection['method']
        self.time = solverSection.as_float('time')
        self.dt = solverSection.as_float('dt')
        self.endtime = solverSection.as_float('endtime')
        self.tolerance = solverSection.as_float('tolerance')
        self.rho_infinity = solverSection.as_float('rho_infinity')
        self.imax = solverSection.as_int('imax')
        self.ci = solverSection.as_float('ci')

        # for solid part
        self.dt_f = solverSection.as_float('dt_f')
        
        self.update_interval = solverSection.as_int('stiffness_update_interval')
        self.constant_pressure = solverSection.as_float('constant_pressure')
        self.constant_T = solverSection.as_float('constant_T')

        self.restart = solverSection.as_bool('restart')
        self.restartFilename = solverSection['restart_displacement_file']
        self.restartTimestep = solverSection.as_int('restart_timestep')

        self.saveStressFilename = None
        self.saveResNum = None


class CVConfig:

    def __init__(self, config):
        """ Walk through the ConfigObj object to
        collect configuration informations. """

        # Constant parameters.
        self.ndim = config.as_int('dimensions')
        self.nSmp = config.as_int('sample_num')
        self.regenerate_samples = config.as_bool('regenerate_samples')

        # Meshes.
        self.meshes = {}
        try:
            meshesConfig = config['meshes']
            for section in meshesConfig.sections:
                if section.startswith('mesh'):
                    self.meshes[meshesConfig[section]['name']] = MeshConfig(meshesConfig[section])
        except KeyError as ex:
            print('Key {} does not exits!'.format(ex))

        # Equation parameters.
        self.equations = {}
        try:
            eqnsConfig = config['equations']
            for section in eqnsConfig.sections:
                if section.startswith('equation'):
                    self.equations[eqnsConfig[section]['name']] = EquationConfig(eqnsConfig[section])
        except KeyError as ex:
            print('Key {} does not exits!'.format(ex))

        # Solver parameters.
        self.solver = SolverConfig(config['solver'])

        # Config the result filename.
        self.solver.saveStressFilename = config['save_stress_filename']
        self.solver.saveResNum = config.as_int('save_result_num')
        # Config the exported stress file name for segregated solver.
        self.solver.exportBdyStressFilename = 'BdyStress'
        if 'export_stress_filename' in config:
            self.solver.exportBdyStressFilename = config['export_stress_filename']

        # Config the number of samples being used in Solid part.
        self.meshes['wall'].nSmp = self.nSmp
        self.meshes['wall'].regenerate_samples = self.regenerate_samples
        self.solver.nSmp = self.nSmp


