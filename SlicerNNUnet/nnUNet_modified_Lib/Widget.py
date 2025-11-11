import traceback
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

import qt
import slicer
from slicer.parameterNodeWrapper import parameterNodeWrapper

from .InstallLogic import InstallLogic, InstallLogicProtocol
from .Parameter import Parameter
from .SegmentationLogic import SegmentationLogic, SegmentationLogicProtocol
from .MorphologyAnalysis import compute_vessel_metrics, save_metrics_to_file


@parameterNodeWrapper
class WidgetParameterNode:
    inputVolume: slicer.vtkMRMLScalarVolumeNode
    parameter: Parameter


class Widget(qt.QWidget):
    """
    nnUNet widget containing an install and run settings collapsible areas.
    Allows to run nnUNet model and displays results in the UI.
    Saves the used settings to QSettings for reloading.
    """

    def __init__(
            self,
            segmentationLogic: Optional[SegmentationLogicProtocol] = None,
            installLogic: Optional[InstallLogicProtocol] = None,
            doShowInfoWindows: bool = True,
            parent=None
    ):
        super().__init__(parent)
        self.logic = segmentationLogic or SegmentationLogic()
        self.installLogic = installLogic or InstallLogic()

        # Instantiate widget UI
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        uiWidget = slicer.util.loadUI(self.resourcePath().joinpath("UI/nnUNet_modified.ui").as_posix())
        uiWidget.setMRMLScene(slicer.mrmlScene)
        layout.addWidget(uiWidget)

        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.ui.inputSelector.currentNodeChanged.connect(self.onInputChanged)
        self.ui.installButton.clicked.connect(self.onInstall)

        self.ui.applyButton.setIcon(self.icon("start_icon.png"))
        self.ui.applyButton.clicked.connect(self.onApply)

        self.ui.stopButton.setIcon(self.icon("stop_icon.png"))
        self.ui.stopButton.clicked.connect(self.onStopClicked)

        # --- Create channel combo box dynamically ---
        self.channelComboBox = qt.QComboBox()
        self.channelComboBox.setToolTip("Select which channel of the volume to display")
        self.channelComboBox.setVisible(False)  # hidden at first

        self.channelLabel = qt.QLabel("Channel:")
        self.channelLabel.setVisible(False)

        channelLayout = qt.QHBoxLayout()
        channelLayout.addWidget(self.channelLabel)
        channelLayout.addWidget(self.channelComboBox)

        self.layout().addLayout(channelLayout)

        # Connect change handler
        self.channelComboBox.currentIndexChanged.connect(self.onChannelChanged)


        # Logic connection
        self.logic.inferenceFinished.connect(self.onInferenceFinished)
        self.logic.errorOccurred.connect(self.onInferenceError)
        self.logic.progressInfo.connect(self.onProgressInfo)
        self.installLogic.progressInfo.connect(self.onProgressInfo)
        self.isStopping = False

        self.sceneCloseObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onSceneChanged)
        self._doShowErrorWindows = doShowInfoWindows

        # Create parameter node and connect GUI
        self._parameterNode = self._createParameterNode()
        self._parameterNode.parameter = Parameter.fromSettings()
        self._parameterNode.connectParametersToGui(
            {
                "parameter.modelPath": self.ui.nnUNetModelPathEdit,
                "parameter.device": self.ui.deviceComboBox,
                "parameter.stepSize": self.ui.stepSizeSlider,
                "parameter.checkPointName": self.ui.checkPointNameLineEdit,
                "parameter.folds": self.ui.foldsLineEdit,
                "parameter.nProcessPreprocessing": self.ui.nProcessPreprocessingSpinBox,
                "parameter.nProcessSegmentationExport": self.ui.nProcessSegmentationExportSpinBox,
                "parameter.disableTta": self.ui.disableTtaCheckBox
            }
        )

        # Configure UI
        self.onInputChanged()
        self.updateInstalledVersion()
        self._setApplyVisible(True)

    @staticmethod
    def _createParameterNode() -> WidgetParameterNode:
        moduleName = "nnUNet_modified"
        parameterNode = slicer.mrmlScene.GetSingletonNode(moduleName, "vtkMRMLScriptedModuleNode")
        if not parameterNode:
            parameterNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
            parameterNode.SetName(slicer.mrmlScene.GenerateUniqueName(moduleName))

        parameterNode.SetAttribute("ModuleName", moduleName)
        return WidgetParameterNode(parameterNode)

    @staticmethod
    def resourcePath() -> Path:
        return Path(__file__).parent.joinpath("..", "Resources")

    @classmethod
    def icon(cls, icon_name) -> "qt.QIcon":
        return qt.QIcon(cls.resourcePath().joinpath("Icons", icon_name).as_posix())

    def __del__(self):
        slicer.mrmlScene.RemoveObserver(self.sceneCloseObserver)
        super().__del__()

    def _setButtonsEnabled(self, isEnabled):
        self.ui.installButton.setEnabled(isEnabled)
        self.ui.applyButton.setEnabled(isEnabled)
        self.ui.inputSelector.setEnabled(isEnabled)

    def onInstall(self, *, doReportFinished=True):
        self._setButtonsEnabled(False)

        if doReportFinished:
            self.ui.logTextEdit.clear()

        success = self.installLogic.setupPythonRequirements(f"nnunetv2{self.ui.toInstallLineEdit.text}")
        if doReportFinished:
            if success:
                self._reportFinished("Install finished correctly.")
            else:
                self._reportError("Install failed.")
        self.updateInstalledVersion()
        self._setButtonsEnabled(True)
        return success

    def _reportError(self, msg, doTraceback=True):
        self.onProgressInfo(msg)
        if self._doShowErrorWindows:
            all_msgs = (msg,) if not doTraceback else (msg, traceback.format_exc())
            slicer.util.errorDisplay(*all_msgs)

    def _reportFinished(self, msg):
        self.onProgressInfo("*" * 80)
        self.onProgressInfo(msg)
        if self._doShowErrorWindows:
            slicer.util.infoDisplay(msg)

    def onLogMessage(self, msg):
        self.ui.logTextEdit.insertPlainText(msg + "\n")

    def updateInstalledVersion(self):
        self.ui.currentVersionLabel.setText(str(self.installLogic.getInstalledNNUnetVersion()))

    def onSceneChanged(self, *_):
        self.onStopClicked()

    def onStopClicked(self):
        self.isStopping = True
        self.logic.stopSegmentation()
        self.logic.waitForSegmentationFinished()
        slicer.app.processEvents()
        self.isStopping = False
        self._setApplyVisible(True)

    def onApply(self, *_):
        if self.getCurrentVolumeNode() is None:
            self._reportError("Please select a valid volume to proceed.")
            return

        self.ui.logTextEdit.clear()
        self.onProgressInfo("Start")
        self.onProgressInfo("*" * 80)

        if not self.onInstall(doReportFinished=False):
            return

        if self.installLogic.needsRestart:
            self._reportFinished("Please restart 3D Slicer to proceed with segmentation.")
            return

        self._setApplyVisible(False)
        self._runSegmentation()

    def _setApplyVisible(self, isVisible):
        self.ui.applyButton.setVisible(isVisible)
        self.ui.stopButton.setVisible(not isVisible)
        self._setButtonsEnabled(isVisible)

    def _runSegmentation(self):
        if self.installLogic.needsRestart:
            self.onInferenceFinished()
            return

        self._parameterNode.parameter.toSettings()
        self.logic.setParameter(self._parameterNode.parameter)
        self.logic.startSegmentation(self.getCurrentVolumeNode())


    @staticmethod
    def detectChannelsFromNode(volumeNode):
        """Return number of channels in a volume, using nibabel if possible."""
        if volumeNode is None:
            return 1  # nothing selected yet

        storageNode = volumeNode.GetStorageNode()
        if not storageNode:
            return 1  # fallback

        filePath = storageNode.GetFullNameFromFileName()
        if not filePath:
            return 1

        try:
            img = nib.load(filePath)
            data = img.get_fdata()
            if data.ndim == 4:  # shape [X, Y, Z, C]
                return data.shape[3]
            else:
                return 1
        except Exception as e:
            print("Nibabel failed:", e)
            return 1

    def onInputChanged(self, *_):

        volumeNode = self.getCurrentVolumeNode()
        self.ui.applyButton.setEnabled(volumeNode is not None)

        # Hide channel controls by default
        self.channelComboBox.clear()
        self.channelComboBox.setVisible(False)
        self.channelLabel.setVisible(False)

        # Reset cache when volume changes
        self._cachedImageData = None

        if not volumeNode:
            return

        # Try to determine the number of channels
        numChannels = 1
        storageNode = volumeNode.GetStorageNode()
        if storageNode:
            filePath = storageNode.GetFullNameFromFileName()
            try:
                import nibabel as nib
                img = nib.load(filePath)
                self._cachedImageData = img.get_fdata()
                numChannels = self._cachedImageData.shape[3] if self._cachedImageData.ndim == 4 else 1
            except Exception as e:
                print("Failed to load with nibabel:", e)

        # Only show channel selector if there are multiple channels
        if numChannels > 1:
            for i in range(numChannels):
                self.channelComboBox.addItem(f"Channel {i}")
            self.channelComboBox.setVisible(True)
            self.channelLabel.setVisible(True)



    def onChannelChanged(self, index):
        if not hasattr(self, "_cachedImageData") or self._cachedImageData is None:
            return  # nothing cached

        if self._cachedImageData.ndim == 4:
            channelData = self._cachedImageData[..., index]
        else:
            channelData = self._cachedImageData  # single channel

        # Convert numpy -> vtkImageData
        channelNode = slicer.util.addVolumeFromArray(channelData.astype(np.float32))
        channelNode.CopyOrientation(self.getCurrentVolumeNode())

        # Reuse preview node if possible
        if not hasattr(self, "_channelPreviewNode"):
            self._channelPreviewNode = channelNode
            self._channelPreviewNode.SetName("ChannelPreview")
            # self._channelPreviewNode.SetHideFromEditors(True)
        else:
            self._channelPreviewNode.SetAndObserveImageData(channelNode.GetImageData())
            slicer.mrmlScene.RemoveNode(channelNode)

        slicer.util.setSliceViewerLayers(background=self._channelPreviewNode)
        slicer.util.resetSliceViews()



    def getCurrentVolumeNode(self):
        return self.ui.inputSelector.currentNode()

    def onInferenceFinished(self, *_):
        if self.isStopping:
            self._setApplyVisible(True)
            return

        try:
            self.onProgressInfo("Loading inference results...")
            segmentation = self.logic.loadSegmentation()
            segmentation.SetName(self.getCurrentVolumeNode().GetName() + "Segmentation")
            self._reportFinished("Inference ended successfully.")

            # --- NEW PART: Morphological analysis ---
            self.onProgressInfo("Running morphological analysis on segmentation...")

            # Convert segmentation to numpy array
            labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentation, labelmap_node, self.getCurrentVolumeNode())
            mask_array = slicer.util.arrayFromVolume(labelmap_node)

            # Compute metrics
            df = compute_vessel_metrics(mask_array)
            csv_path = save_metrics_to_file(df)

            # Notify user
            self._reportFinished(f"Morphological analysis complete.\nResults saved to:\n{csv_path}")
            slicer.util.openAddDataDialog([csv_path])

        except RuntimeError as e:
            self._reportError(f"Inference ended in error:\n{e}")
        finally:
            self._setApplyVisible(True)

    def onInferenceError(self, errorMsg):
        if self.isStopping:
            return

        self._setApplyVisible(True)
        if isinstance(errorMsg, Exception):
            errorMsg = str(errorMsg)
        self._reportError("Encountered error during inference :\n" + errorMsg, doTraceback=False)

    def onProgressInfo(self, infoMsg):
        self.ui.logTextEdit.insertPlainText(self._formatMsg(infoMsg) + "\n")
        self.moveTextEditToEnd(self.ui.logTextEdit)
        slicer.app.processEvents()

    @staticmethod
    def _formatMsg(infoMsg):
        return "\n".join([msg for msg in infoMsg.strip().splitlines()])

    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)
