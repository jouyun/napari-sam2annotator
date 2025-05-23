from typing import TYPE_CHECKING
from typing import Any, Generator, Optional

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLineEdit, QVBoxLayout, QFrame, QLabel, QFileDialog, QCheckBox
import numpy as np
import skimage.util as util
import tifffile
import skimage.data as data
import os
import time
import glob
import cv2
import tifffile
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from matplotlib.widgets import LassoSelector, RectangleSelector, SpanSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from qtpy.QtWidgets import QComboBox, QCompleter
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QListWidget
from qtpy.QtWidgets import QFileDialog
import skimage as ski
import napari_sam2annotator._sam2_wrapper as _sam2_wrapper
from ._sam2_wrapper import sam2_wrapper
from ._funcs import delete_tmp_files, convert_to_rgb, cleanup_mask, keep_largest, is_legit_shape

if TYPE_CHECKING:
    import napari
    
class sam2annotatorWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.root_directory = ''

        self.figure = Figure()

        # Create the buttons
        self.zoom_combo = QComboBox()
        for factor in (1, 2, 4, 8, 16):
            self.zoom_combo.addItem(f"{factor}X", userData=factor)  # store numeric value
        self.zoom_combo.setCurrentIndex(0)  # set default to 1X

        self.folder_btn = QPushButton("folder")
        self.folder_btn.clicked.connect(self._folder_click)
        self.open_btn = QPushButton("open")
        self.open_btn.clicked.connect(self._open_click)

        self.set_btn = QPushButton("set")
        self.set_btn.clicked.connect(self._set_click)

        self.multibox_check = QCheckBox("Multi Box")

        self.box_btn = QPushButton("box")
        self.box_btn.clicked.connect(self._box_click)


        # Internal variables
        self.fnames = []
        self.file_index = 0
        self.base_layer = None
        self.proccessing_directory = False
        
        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.zoom_combo)
        layout.addWidget(self.folder_btn)
        layout.addWidget(self.open_btn)
        layout.addWidget(self.set_btn)
        layout.addWidget(self.multibox_check)
        layout.addWidget(self.box_btn)

        # Finish the layout
        self.setLayout(layout)
        self.sam = sam2_wrapper()
        self.viewer.bind_key("Shift-C", self._on_c_click)
        self.viewer.bind_key("Shift-A", self._on_a_click)
        self.viewer.bind_key("Shift-D", self._on_d_click)
        self.viewer.bind_key("Shift-T", self._on_t_click)
        self.viewer.bind_key("Shift-B", self._on_b_click)

    # Button presses        
    def _folder_click(self):
        path = QFileDialog.getExistingDirectory(
            parent=self,                           # modal to the widget
            caption="Select folder",
        )
        if path:                                   # user didn’t cancel
            self.proccessing_directory = True
            self.root_directory = path
            self.fnames = glob.glob(os.path.join(path, '*.tif'))
            self.fnames.sort()
            self.file_index = self.get_next_undone(0)
            if self.file_index >= len(self.fnames):
                print('No more files to process')
                return
            self.file_name = self.fnames[self.file_index]
            self.open_image(self.file_name)

    def _open_click(self):
        path, _ = QFileDialog.getOpenFileName(
            parent=self,                           # modal to the widget
            caption="Open image",
            filter="Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)",
        )
        if path:                                   # user didn’t cancel
            self.root_directory = os.path.dirname(path)
            self.file_name = os.path.basename(path)
            self.open_image(path)

    def _set_click(self):
        if self.base_layer is None:
            self.annotated_img = self.viewer.layers.selection.active.data
            cur_name = self.viewer.layers.selection.active.name
            cur_idx = -1
            for id, layer in enumerate(self.viewer.layers):
                if layer.name == cur_name:
                    cur_idx = id
                    break
            self.base_layer = cur_idx
        else:
            self.annotated_img = self.viewer.layers[self.base_layer].data
        self.sam.set_image(self.annotated_img)
        self.clear_boxes()
    
    def _box_click(self):
        shapes = self.viewer.layers['Box'].data[1:]
        if self.multibox_check.isChecked():
            mask = self.sam.infer_from_box(shapes, do_reset=False)
        else:
            mask = self.sam.infer_from_box_single_object(shapes)
        if 'Labels' in self.viewer.layers:
            self.viewer.layers['Labels'].data = mask.astype(int)*3
        else:
            self.viewer.add_labels(mask.astype(int)*3)
        self.clear_boxes()


    # Utility functions
    def open_image(self, path):
        downsample = self.zoom_combo.currentIndex()
        downsample = 2**downsample
        print(f"downsample: {downsample}")
        self.img = ski.io.imread(path)
        # If one of the dimensions is 5 or less, assume it is channels
        channel_axis = np.argmin(self.img.shape)
        if self.img.shape[channel_axis] <= 5:
            self.img = np.moveaxis(self.img, channel_axis, 0)
            self.orig_shape = self.img.shape[1:]
            self.img = ski.transform.downscale_local_mean(self.img, (1,1,downsample, downsample))
            self.viewer.add_image(self.img, name=os.path.basename(path), channel_axis=0)
        else:
            self.orig_shape = self.img.shape
            self.img = ski.transform.downscale_local_mean(self.img, (1, downsample, downsample))
            self.viewer.add_image(self.img)

    def clear_boxes(self):
        if 'Box' in self.viewer.layers:
            self.viewer.layers.remove('Box')
        dummy_shape = np.array([[0,0,0], [0,0,1], [0, 1, 1], [0,1,0]])
        self.viewer.add_shapes([dummy_shape], ndim=3, name='Box')

    def upsample_stack_cv2(self, img, y_dim, x_dim):
        # Upsample the image using OpenCV
        rtn_img = []
        for i in range(img.shape[0]):
            rtn_img.append(cv2.resize(img[i], (x_dim, y_dim), interpolation=cv2.INTER_NEAREST))
        return np.array(rtn_img)
    
    def get_next_undone(self, file_index):
        if file_index >= len(self.fnames):
            return file_index
        file_name = self.fnames[file_index]
        out_name = file_name.replace('.tif', '_labels.tiff')

        while os.path.exists(out_name):
            file_index += 1
            if file_index >= len(self.fnames):
                break
            file_name = self.fnames[file_index]
            out_name = file_name.replace('.tif', '_labels.tiff')
        return file_index

    # Key bindings
    def _on_c_click(self, _: Optional[Any] = None) -> None:
        self._box_click()
    
    def _on_a_click(self, _: Optional[Any] = None) -> None:
        current_mask = self.viewer.layers['Labels'].data
        # Keep only the largest contiguous object
        current_mask = keep_largest(current_mask)
        if 'master_labels' in self.viewer.layers:
            self.viewer.layers['master_labels'].data[current_mask>0] = np.max(self.viewer.layers['master_labels'].data) + 1
        else:
            self.viewer.add_labels(current_mask, name='master_labels')
        self.viewer.layers.selection.active = self.viewer.layers['Box']

    def _on_t_click(self, _: Optional[Any] = None) -> None:
        current_slice = self.viewer.dims.current_step[0]
        current_labels = self.viewer.layers['Labels'].data
        current_labels[current_slice:] = 0
        self.viewer.layers['Labels'].data = current_labels
    
    def _on_b_click(self, _: Optional[Any] = None) -> None:
        current_slice = self.viewer.dims.current_step[0]
        current_labels = self.viewer.layers['Labels'].data
        current_labels[:current_slice] = 0
        self.viewer.layers['Labels'].data = current_labels

    def _on_d_click(self, _: Optional[Any] = None) -> None:
        out_data = self.viewer.layers['master_labels'].data
        out_data = self.upsample_stack_cv2(out_data, self.orig_shape[1], self.orig_shape[2])
        ski.io.imsave(os.path.join(self.root_directory, self.file_name.replace('.tif', '_labels.tiff')), out_data.astype(np.uint16))
        if self.proccessing_directory:
            self.file_index += 1
            self.file_index = self.get_next_undone(self.file_index)
            if len(self.fnames) > self.file_index:
                self.file_name = self.fnames[self.file_index]
                self.viewer.layers.clear()
                self.open_image(self.file_name)
                self._set_click()