#!/usr/bin/env python3
"""
GUI interface for PoseMapper with file picker dialogs
"""

import sys
import os
from pathlib import Path
import subprocess

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
                             QGroupBox, QFileDialog, QMessageBox, QScrollArea,
                             QFrame, QGridLayout, QTextEdit, QDialog, QDialogButtonBox,
                             QProgressBar, QTextBrowser)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QProcess
from PyQt5.QtGui import QFont, QPalette, QColor, QCloseEvent

class PoseMapperWorker(QThread):
    """Worker thread to run PoseMapper using QProcess"""
    progress_updated = pyqtSignal(str)  # Signal for status updates
    finished_success = pyqtSignal()
    finished_error = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None

    def run(self):
        """Run the PoseMapper process in a separate thread"""
        try:
            self.process = QProcess()
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.finished.connect(self.handle_finished)

            # Set working directory to the PoseMapper directory
            import os
            working_dir = os.path.dirname(os.path.abspath(__file__))
            self.process.setWorkingDirectory(working_dir)

            # Start the process
            program = self.command[0]
            arguments = self.command[1:]

            # Debug: print what we're trying to run
            cmd_str = program + " " + " ".join(arguments)
            self.progress_updated.emit(f"Starting command: {cmd_str}")
            self.progress_updated.emit(f"Working directory: {working_dir}")

            self.process.start(program, arguments)

            if not self.process.waitForStarted(5000):  # 5 second timeout
                error = self.process.errorString()
                self.progress_updated.emit(f"Process error: {error}")
                self.finished_error.emit(f"Failed to start process: {error}")
                return

            self.progress_updated.emit("Process started successfully")

            # Keep the thread alive while process is running
            self.exec_()

        except Exception as e:
            self.finished_error.emit(f"Exception in worker thread: {str(e)}")

    def handle_stdout(self):
        """Handle stdout output"""
        if self.process:
            data = self.process.readAllStandardOutput()
            output = bytes(data).decode('utf-8', errors='ignore')
            for line in output.split('\n'):
                line = line.strip()
                if line:
                    self.progress_updated.emit(line)

    def handle_stderr(self):
        """Handle stderr output"""
        if self.process:
            data = self.process.readAllStandardError()
            output = bytes(data).decode('utf-8', errors='ignore')
            for line in output.split('\n'):
                line = line.strip()
                if line:
                    self.progress_updated.emit(line)

    def handle_finished(self, exit_code, exit_status):
        """Handle process completion"""
        if exit_code == 0:
            self.finished_success.emit()
        else:
            self.finished_error.emit(f"Process exited with code {exit_code}")
        # Exit the thread's event loop
        self.quit()

    def stop(self):
        """Stop the process if running"""
        if self.process and self.process.state() == QProcess.Running:
            self.process.kill()
            self.process.waitForFinished(3000)  # Wait up to 3 seconds

class PoseMapperGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.original_title = "PoseMapper - Easy Mode"
        self.setWindowTitle(self.original_title)
        self.setGeometry(100, 100, 900, 800)
        self.setMinimumSize(800, 600)

        # Create scroll area for better layout
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create main content widget
        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)

        # Content layout
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(10)
        self.content_layout.setContentsMargins(20, 20, 20, 20)

        # Initialize variables
        self.input_file = ""
        self.output_file = ""
        self.character_image = ""
        self.model_type = "COCO"
        self.style = "default"
        self.background = ""
        self.show_keypoints = False
        self.show_confidence = False
        self.show_angles = False
        self.use_gpu = False
        self.export_json = False
        self.json_file = ""
        self.no_display = False
        self.fps = ""
        self.resolution = ""

        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_label = QLabel("PoseMapper - Easy Mode")
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(title_label)

        # File Selection Group
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout(file_group)
        file_layout.setSpacing(10)

        # Input File
        input_label = QLabel("Input Video:")
        self.input_entry = QLineEdit()
        self.input_entry.setPlaceholderText("Select input video file...")
        self.input_browse_btn = QPushButton("Browse...")
        self.input_browse_btn.clicked.connect(self.browse_input)

        file_layout.addWidget(input_label, 0, 0)
        file_layout.addWidget(self.input_entry, 0, 1)
        file_layout.addWidget(self.input_browse_btn, 0, 2)

        # Output File
        output_label = QLabel("Output Video:")
        self.output_entry = QLineEdit()
        self.output_entry.setPlaceholderText("Select output video file...")
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output)

        file_layout.addWidget(output_label, 1, 0)
        file_layout.addWidget(self.output_entry, 1, 1)
        file_layout.addWidget(self.output_browse_btn, 1, 2)

        # Character Image
        character_label = QLabel("Character Image:")
        self.character_entry = QLineEdit()
        self.character_entry.setPlaceholderText("Select character image (optional)...")
        self.character_browse_btn = QPushButton("Browse...")
        self.character_browse_btn.clicked.connect(self.browse_character)

        file_layout.addWidget(character_label, 2, 0)
        file_layout.addWidget(self.character_entry, 2, 1)
        file_layout.addWidget(self.character_browse_btn, 2, 2)

        self.content_layout.addWidget(file_group)

        # Model Settings Group
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout(model_group)

        # Model Type
        model_type_layout = QHBoxLayout()
        model_type_label = QLabel("Model Type:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["COCO", "BODY_25", "MPI"])
        self.model_combo.setCurrentText(self.model_type)
        self.model_combo.currentTextChanged.connect(self.update_model_type)

        model_type_layout.addWidget(model_type_label)
        model_type_layout.addWidget(self.model_combo)
        model_type_layout.addStretch()
        model_layout.addLayout(model_type_layout)

        self.content_layout.addWidget(model_group)

        # Rendering Options Group
        render_group = QGroupBox("Rendering Options")
        render_layout = QGridLayout(render_group)
        render_layout.setSpacing(10)

        # Style
        style_label = QLabel("Style:")
        self.style_combo = QComboBox()
        self.style_combo.addItems(["default", "glow", "neon", "minimal"])
        self.style_combo.setCurrentText(self.style)
        self.style_combo.currentTextChanged.connect(self.update_style)

        render_layout.addWidget(style_label, 0, 0)
        render_layout.addWidget(self.style_combo, 0, 1)

        # Background
        background_label = QLabel("Background:")
        self.background_combo = QComboBox()
        self.background_combo.addItems(["Original Video", "Black", "White"])
        self.background_combo.setCurrentText("Original Video")
        self.background_combo.currentTextChanged.connect(self.update_background)

        render_layout.addWidget(background_label, 1, 0)
        render_layout.addWidget(self.background_combo, 1, 1)

        # Checkboxes
        self.keypoints_check = QCheckBox("Show Keypoint Names")
        self.keypoints_check.stateChanged.connect(self.update_show_keypoints)
        render_layout.addWidget(self.keypoints_check, 2, 0)

        self.confidence_check = QCheckBox("Show Confidence Scores")
        self.confidence_check.stateChanged.connect(self.update_show_confidence)
        render_layout.addWidget(self.confidence_check, 2, 1)

        self.angles_check = QCheckBox("Show Joint Angles")
        self.angles_check.stateChanged.connect(self.update_show_angles)
        render_layout.addWidget(self.angles_check, 3, 0)

        self.gpu_check = QCheckBox("Use GPU (if available)")
        self.gpu_check.stateChanged.connect(self.update_use_gpu)
        render_layout.addWidget(self.gpu_check, 3, 1)

        self.content_layout.addWidget(render_group)

        # Export Options Group
        export_group = QGroupBox("Export Options")
        export_layout = QGridLayout(export_group)
        export_layout.setSpacing(10)

        self.json_check = QCheckBox("Export Pose Data to JSON")
        self.json_check.stateChanged.connect(self.toggle_json_entry)
        export_layout.addWidget(self.json_check, 0, 0)

        self.json_entry = QLineEdit()
        self.json_entry.setPlaceholderText("Select JSON file...")
        self.json_entry.setEnabled(False)
        export_layout.addWidget(self.json_entry, 0, 1)

        self.json_browse_btn = QPushButton("Browse...")
        self.json_browse_btn.clicked.connect(self.browse_json)
        export_layout.addWidget(self.json_browse_btn, 0, 2)

        self.content_layout.addWidget(export_group)

        # Advanced Options Group
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QGridLayout(advanced_group)
        advanced_layout.setSpacing(10)

        # FPS
        fps_label = QLabel("Output FPS (optional):")
        self.fps_entry = QLineEdit()
        self.fps_entry.setPlaceholderText("e.g., 30")
        self.fps_entry.textChanged.connect(self.update_fps)

        advanced_layout.addWidget(fps_label, 0, 0)
        advanced_layout.addWidget(self.fps_entry, 0, 1)

        # Resolution
        resolution_label = QLabel("Resolution (WxH, optional):")
        self.resolution_entry = QLineEdit()
        self.resolution_entry.setPlaceholderText("e.g., 1920x1080")
        self.resolution_entry.textChanged.connect(self.update_resolution)

        advanced_layout.addWidget(resolution_label, 1, 0)
        advanced_layout.addWidget(self.resolution_entry, 1, 1)

        # No Display
        self.no_display_check = QCheckBox("No Display (headless mode)")
        self.no_display_check.stateChanged.connect(self.update_no_display)
        advanced_layout.addWidget(self.no_display_check, 2, 0, 1, 2)

        self.content_layout.addWidget(advanced_group)

        # Progress and Status Section
        progress_group = QGroupBox("Processing Status")
        progress_layout = QVBoxLayout(progress_group)

        self.status_text = QTextBrowser()
        self.status_text.setMaximumHeight(150)
        self.status_text.setPlainText("Ready to process...")
        progress_layout.addWidget(self.status_text)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        # Clear button for status
        clear_status_layout = QHBoxLayout()
        self.clear_status_button = QPushButton("Clear Status")
        self.clear_status_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 12px;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
        """)
        self.clear_status_button.clicked.connect(self.clear_status)
        clear_status_layout.addStretch()
        clear_status_layout.addWidget(self.clear_status_button)
        clear_status_layout.addStretch()
        progress_layout.addLayout(clear_status_layout)

        self.content_layout.addWidget(progress_group)

        # Add stretch to push buttons to bottom
        self.content_layout.addStretch()

        # Action Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.run_button = QPushButton("Run PoseMapper")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.run_button.clicked.connect(self.run_posemapper)

        self.show_command_button = QPushButton("Show Command")
        self.show_command_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                padding: 12px 24px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        self.show_command_button.clicked.connect(self.show_command)

        self.exit_button = QPushButton("Exit")
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                padding: 12px 24px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.exit_button.clicked.connect(self.exit_application)

        button_layout.addStretch()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.show_command_button)
        button_layout.addWidget(self.exit_button)
        button_layout.addStretch()

        self.content_layout.addLayout(button_layout)

    def disable_all_controls(self):
        """Disable all input controls during processing"""
        # Update window title to show processing status
        self.setWindowTitle(f"{self.original_title} - PROCESSING...")

        # File selection controls
        self.input_entry.setEnabled(False)
        self.output_entry.setEnabled(False)
        self.character_entry.setEnabled(False)
        self.input_browse_btn.setEnabled(False)
        self.output_browse_btn.setEnabled(False)
        self.character_browse_btn.setEnabled(False)

        # Model settings
        self.model_combo.setEnabled(False)

        # Rendering options
        self.style_combo.setEnabled(False)
        self.background_combo.setEnabled(False)
        self.keypoints_check.setEnabled(False)
        self.confidence_check.setEnabled(False)
        self.angles_check.setEnabled(False)
        self.gpu_check.setEnabled(False)

        # Export options
        self.json_check.setEnabled(False)
        self.json_entry.setEnabled(False)
        self.json_browse_btn.setEnabled(False)

        # Advanced options
        self.fps_entry.setEnabled(False)
        self.resolution_entry.setEnabled(False)
        self.no_display_check.setEnabled(False)

        # Action buttons
        self.run_button.setEnabled(False)
        self.show_command_button.setEnabled(False)
        # Keep exit button enabled for emergency exit

    def enable_all_controls(self):
        """Re-enable all input controls after processing"""
        # Restore original window title
        self.setWindowTitle(self.original_title)

        # File selection controls
        self.input_entry.setEnabled(True)
        self.output_entry.setEnabled(True)
        self.character_entry.setEnabled(True)
        self.input_browse_btn.setEnabled(True)
        self.output_browse_btn.setEnabled(True)
        self.character_browse_btn.setEnabled(True)

        # Model settings
        self.model_combo.setEnabled(True)

        # Rendering options
        self.style_combo.setEnabled(True)
        self.background_combo.setEnabled(True)
        self.keypoints_check.setEnabled(True)
        self.confidence_check.setEnabled(True)
        self.angles_check.setEnabled(True)
        self.gpu_check.setEnabled(True)

        # Export options
        self.json_check.setEnabled(True)
        # JSON entry enabled state depends on checkbox
        self.json_entry.setEnabled(self.export_json)
        self.json_browse_btn.setEnabled(True)

        # Advanced options
        self.fps_entry.setEnabled(True)
        self.resolution_entry.setEnabled(True)
        self.no_display_check.setEnabled(True)

        # Action buttons
        self.run_button.setEnabled(True)
        self.show_command_button.setEnabled(True)

    def toggle_json_entry(self, state):
        """Enable/disable JSON file entry based on checkbox"""
        self.export_json = state == 2  # Qt.CheckState.Checked
        self.json_entry.setEnabled(self.export_json)

    def update_model_type(self, text):
        self.model_type = text

    def update_style(self, text):
        self.style = text

    def update_background(self, text):
        if text == "Original Video":
            self.background = ""
        elif text == "Black":
            self.background = "black"
        elif text == "White":
            self.background = "white"

    def update_show_keypoints(self, state):
        self.show_keypoints = state == 2

    def update_show_confidence(self, state):
        self.show_confidence = state == 2

    def update_show_angles(self, state):
        self.show_angles = state == 2

    def update_use_gpu(self, state):
        self.use_gpu = state == 2

    def update_no_display(self, state):
        self.no_display = state == 2

    def update_fps(self, text):
        self.fps = text

    def update_resolution(self, text):
        self.resolution = text
    
    def clear_status(self):
        """Clear the status text and reset progress bar"""
        self.status_text.clear()
        self.status_text.append("Status cleared. Ready to process...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
    
    def browse_input(self):
        """Browse for input video file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*.*)")
        if filename:
            self.input_file = filename
            self.input_entry.setText(filename)
            # Auto-generate output filename
            if not self.output_file:
                input_path = Path(filename)
                output_path = input_path.parent / f"{input_path.stem}_with_pose{input_path.suffix}"
                self.output_file = str(output_path)
                self.output_entry.setText(self.output_file)

    def browse_output(self):
        """Browse for output video file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", "",
            "MP4 files (*.mp4);;AVI files (*.avi);;All files (*.*)")
        if filename:
            self.output_file = filename
            self.output_entry.setText(filename)

    def browse_json(self):
        """Browse for JSON export file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Pose Data", "",
            "JSON files (*.json);;All files (*.*)")
        if filename:
            self.json_file = filename
            self.json_entry.setText(filename)

    def browse_character(self):
        """Browse for character image file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Character Image", "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff);;All files (*.*)")
        if filename:
            self.character_image = filename
            self.character_entry.setText(filename)
    
    def build_command(self):
        """Build the command line arguments"""
        import sys
        cmd = [sys.executable, "main.py"]

        if self.input_file:
            cmd.extend(["--input", self.input_file])

        if self.character_image:
            cmd.extend(["--character-image", self.character_image])

        if self.output_file:
            cmd.extend(["--output", self.output_file])

        cmd.extend(["--model-type", self.model_type])
        cmd.extend(["--style", self.style])

        if self.background:
            cmd.extend(["--background", self.background])

        if self.show_keypoints:
            cmd.append("--show-keypoint-names")

        if self.show_confidence:
            cmd.append("--show-confidence")

        if self.show_angles:
            cmd.append("--show-angles")

        if self.use_gpu:
            cmd.append("--use-gpu")

        if self.export_json and self.json_file:
            cmd.extend(["--export-json", self.json_file])

        if self.no_display:
            cmd.append("--no-display")

        if self.fps:
            cmd.extend(["--fps", self.fps])

        if self.resolution:
            cmd.extend(["--resolution", self.resolution])

        return cmd
    
    def show_command(self):
        """Show the command that will be executed"""
        cmd = self.build_command()
        command_str = " ".join(cmd)

        # Create a dialog to show the command
        dialog = QDialog(self)
        dialog.setWindowTitle("Command to Execute")
        dialog.setFixedSize(700, 200)

        layout = QVBoxLayout(dialog)

        label = QLabel("Command to execute:")
        label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(label)

        text_edit = QTextEdit()
        text_edit.setPlainText(command_str)
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Courier New", 10))
        layout.addWidget(text_edit)

        button_layout = QHBoxLayout()

        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(command_str))
        button_layout.addWidget(copy_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        dialog.exec_()
    
    def run_posemapper(self):
        """Run PoseMapper with the selected options"""
        # Validate inputs
        if not self.input_file:
            QMessageBox.critical(self, "Error", "Please select an input video file.")
            return

        if not self.output_file:
            QMessageBox.critical(self, "Error", "Please specify an output video file.")
            return

        if self.export_json and not self.json_file:
            QMessageBox.critical(self, "Error", "Please specify a JSON file for export.")
            return

        # Confirm and run
        cmd = self.build_command()
        command_str = " ".join(cmd)

        reply = QMessageBox.question(self, "Confirm",
                                   f"Run PoseMapper with the following command?\n\n{command_str}",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Disable all controls during processing
            self.disable_all_controls()
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_text.clear()
            self.status_text.append("Starting PoseMapper...")

            # Create and start worker thread
            self.worker = PoseMapperWorker(cmd)

            # Connect signals
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.finished_success.connect(self.on_success)
            self.worker.finished_error.connect(self.on_error)
            self.worker.finished.connect(self.worker.deleteLater)

            # Start the worker thread
            self.worker.start()

    def update_progress(self, message):
        """Update progress display with new message"""
        self.status_text.append(message)

        # Try to extract progress percentage from message
        if "Progress:" in message:
            try:
                # Extract percentage from "Progress: X.X%" format
                progress_str = message.split("Progress:")[1].strip()
                percentage = float(progress_str.rstrip('%'))
                self.progress_bar.setValue(int(percentage))
            except (ValueError, IndexError):
                pass

    def on_success(self):
        """Handle successful completion"""
        self.progress_bar.setValue(100)
        self.status_text.append("PoseMapper completed successfully!")
        self.cleanup_after_processing()
        QMessageBox.information(self, "Success", "PoseMapper completed successfully!")

    def on_error(self, error_msg):
        """Handle processing error"""
        self.status_text.append(f"Error: {error_msg}")
        self.cleanup_after_processing()
        QMessageBox.critical(self, "Error", f"Error running PoseMapper:\n{error_msg}")

    def cleanup_after_processing(self):
        """Re-enable all controls and hide progress bar after processing"""
        self.enable_all_controls()

        # Stop and cleanup worker thread
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
            if self.worker.isRunning():
                self.worker.wait(3000)  # Wait up to 3 seconds

        # Keep progress bar visible for a moment to show completion
        QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))

    def exit_application(self):
        """Exit the application with proper cleanup"""
        # Stop any running worker thread
        if hasattr(self, 'worker') and self.worker:
            self.status_text.append("Stopping running process...")
            self.worker.stop()
            if self.worker.isRunning():
                self.worker.wait(5000)  # Wait up to 5 seconds for graceful shutdown

        # Close the application
        self.close()

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event (X button, Alt+F4, etc.)"""
        # Stop any running worker thread before closing
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
            if self.worker.isRunning():
                self.worker.wait(5000)  # Wait up to 5 seconds for graceful shutdown

        # Accept the close event
        event.accept()

def main():
    """Main function to run the GUI"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    # Set application properties
    app.setApplicationName("PoseMapper")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("PoseMapper")

    window = PoseMapperGUI()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()