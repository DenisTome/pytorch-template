# -*- coding: utf-8 -*-
"""
Viwer class to be called specifying the callback function as well
as the draw and imgui functions.

@author: Carsten Stoll (extended by Denis Tome)

Copyright Epic Games, Inc. All Rights Reserved.

"""

import pygame as pg
import OpenGL.GL as gl
import OpenGL.GLU as glu
import numpy as np

from imgui.integrations.pygame import PygameRenderer
import imgui


class Viewer:
    """Viewer class"""

    def __init__(self):
        """Init"""

        self._current_frame = 0
        self._camera_pre_rotation = np.array((0.0, 0.0, 0.0))
        self._camera_center = np.array((0.0, 0.0, 0.0))
        self._camera_angles = np.array((0.0, 0.0))
        self._camera_distance = -250.0

        pg.init()
        pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1)
        pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, 16)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)

        self._window_size = (1280, 720)
        self._surface = pg.display.set_mode(
            self._window_size,
            pg.RESIZABLE | pg.DOUBLEBUF | pg.OPENGL)

        self._running = True
        self._last_mouse_pos = np.array((0, 0))
        self._last_mouse_button = -1
        self._last_mouse_button_time = pg.time.get_ticks()
        self._clock = pg.time.Clock()

        # to be defined by the user
        self._draw_function = lambda frame, clock: ()
        self._key_callback = lambda key: ()
        self._imgui_function = lambda: ()
        self._show_imgui = False
        self._is_draw_defined = False

        # enable default opengl settings
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)

        # set up lights
        gl.glShadeModel(gl.GL_SMOOTH)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, [20.0])
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK,
                        gl.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

        # set up material
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glColor3f(1.0, 1.0, 1.0)

        self._modelview = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)

        # create imgui objects
        imgui.create_context()
        self._imgui_impl = PygameRenderer()
        self._imgui_io = imgui.get_io()

    def set_key_callback(self, funct):
        """Set callback function"""
        self._key_callback = funct

    def set_imgui_function(self, funct):
        """Set imgui function"""
        self._show_imgui = True
        self._imgui_function = funct

    def set_draw_function(self, funct):
        """Set callback function"""
        self._is_draw_defined = True
        self._draw_function = funct

    def _check_events(self, event):
        """Check events

        Args:
            event (pg.event): event
        """

        if event.type != pg.VIDEORESIZE:
            self._imgui_impl.process_event(event)
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self._running = False
            else:
                self._key_callback(chr(event.key))
        elif event.type == pg.QUIT:
            self._running = False
        elif event.type == pg.MOUSEMOTION and not self._imgui_io.want_capture_mouse:
            # check for mouse motion
            pos = np.array(pg.mouse.get_pos())
            diff = pos - self._last_mouse_pos
            self._last_mouse_pos = pos
            if pg.mouse.get_pressed()[0]:
                # rotation
                self._camera_angles += diff * 0.5
                self._camera_angles[1] = np.clip(
                    self._camera_angles[1], -90.0, 90.0)
            elif pg.mouse.get_pressed()[1]:
                scale = -0.0012 * self._camera_distance
                self._camera_center -= diff[0] * \
                    self._modelview[0:3, 0] * scale
                self._camera_center += diff[1] * \
                    self._modelview[0:3, 1] * scale
        elif event.type == pg.MOUSEBUTTONUP and not self._imgui_io.want_capture_mouse:
            # check for mouse button up for double click
            self._last_mouse_button = event.button
            self._last_mouse_button_time = pg.time.get_ticks()
        elif event.type == pg.MOUSEBUTTONDOWN and not self._imgui_io.want_capture_mouse:
            # check for double-click
            if event.button == 1 and self._last_mouse_button == 1:
                t = pg.time.get_ticks() - self._last_mouse_button_time
                if t < 120:
                    # double click, focus on point under mouse
                    x = event.pos[0]
                    y = self._window_size[1] - event.pos[1] - 1
                    depth = gl.glReadPixelsf(
                        x, y, 1, 1, gl.GL_DEPTH_COMPONENT)
                    if depth < 1.0:
                        self._camera_center = glu.gluUnProject(
                            x, y, depth)
            elif event.button == 4:
                # mouse wheel
                self._camera_distance *= 0.9
            elif event.button == 5:
                # mouse wheel
                self._camera_distance /= 0.9
        elif event.type == pg.VIDEORESIZE:
            # resize the viewport
            self._window_size = event.size
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            glu.gluPerspective(
                45, (self._window_size[0] / self._window_size[1]), 1.0, 1000.0)
            gl.glViewport(
                0, 0, self._window_size[0], self._window_size[1])
            gl.glGetFloatv(gl.GL_PROJECTION_MATRIX)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            self._imgui_io.display_size = self._window_size

    def run(self):
        """Run"""

        if not self._is_draw_defined:
            raise RuntimeError('Please define draw function!')

        while self._running:
            for event in pg.event.get():
                self._check_events(event)

            # process imgui data
            if self._show_imgui:
                imgui.new_frame()
                imgui.show_test_window()
                self._imgui_function()

            # clear buffer
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # enable opengl settings for rendering
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_COLOR_MATERIAL)

            # set up camera correctly
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.5, 0.5, 1.0, 0.0])
            gl.glTranslatef(0.0, 0.0, self._camera_distance)
            gl.glRotatef(self._camera_angles[1], 1, 0, 0)
            gl.glRotatef(self._camera_angles[0], 0, 1, 0)
            gl.glRotatef(self._camera_pre_rotation[2], 0, 0, 1)
            gl.glRotatef(self._camera_pre_rotation[1], 0, 1, 0)
            gl.glRotatef(self._camera_pre_rotation[0], 1, 0, 0)
            gl.glTranslatef(-self._camera_center[0], -
                            self._camera_center[1], -self._camera_center[2])
            self._modelview = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)

            # call the draw function
            self._draw_function(self._current_frame, self._clock.get_time())

            # draw ui on top
            if self._show_imgui:
                gl.glDisable(gl.GL_LIGHTING)
                gl.glDisable(gl.GL_LIGHT0)
                imgui.render()
                self._imgui_impl.render(imgui.get_draw_data())

            # swap double buffer
            self._clock.tick(60.0)
            pg.display.flip()

    pg.quit()
