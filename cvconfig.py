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
            print('Key {} does not exits!'.format(ex))


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

