"""Application entry point: single-instance enforcement, startup dialogs, and the main window."""
import os
import sys

# A windowed (console=False) PyInstaller build has no stdout/stderr handles at all,
# so any stray print() would raise instead of silently doing nothing.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

video_arg = None
if len(sys.argv) >= 2:
    video_arg = os.path.abspath(sys.argv[1])

if hasattr(sys, '_MEIPASS'):
    os.chdir(sys._MEIPASS)
    os.environ["QT_PLUGIN_PATH"] = os.path.join(sys._MEIPASS, 'PySide6', 'plugins')

from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog, QWidget, QDialog
from PySide6.QtCore import QSettings, QCoreApplication, QTimer, Qt
from PySide6.QtNetwork import QLocalSocket, QLocalServer
from auth import load_api_keys, save_api_keys
from ui_components import ApiKeyDialog
from main_window import VideoPlayerWidget

IPC_SERVER_NAME = "Cinevoice_SingleInstance_Server"

def main():
    if hasattr(sys, '_MEIPASS'):
        plugin_path = os.path.join(sys._MEIPASS, 'PySide6', 'plugins')
        QCoreApplication.addLibraryPath(plugin_path)

    app = QApplication(sys.argv)
    app.setOrganizationName("amiri.moalla@gmail.com")
    app.setApplicationName("Cinevoice")
    
    # Another instance is already running: hand it the requested file (if any) and exit,
    # so double-clicking a video always reuses the same window instead of opening a new one.
    socket = QLocalSocket()
    socket.connectToServer(IPC_SERVER_NAME)
    if socket.waitForConnected(500):
        if video_arg:
            socket.write(video_arg.encode('utf-8'))
            socket.waitForBytesWritten(1000)
        sys.exit(0)
    
    server = QLocalServer()
    server.removeServer(IPC_SERVER_NAME)
    if not server.listen(IPC_SERVER_NAME):
        pass

    q_settings = QSettings()
    dummy_parent = QWidget()
    
    api_keys = load_api_keys()

    from local_vlm import get_downloaded_local_models
    if not any(api_keys.values()) and not get_downloaded_local_models():
       dialog = ApiKeyDialog(api_keys, dummy_parent)
       if dialog.exec() == QDialog.DialogCode.Accepted:
           new_keys = dialog.get_keys()
           save_api_keys(new_keys)
           api_keys = load_api_keys()
               
    path = video_arg
    if not path:
       last_dir = q_settings.value("last_directory", ".")
       path, _ = QFileDialog.getOpenFileName(
          dummy_parent, "Select video file", last_dir, "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*.*)"
       )
       if not path:
          return
       q_settings.setValue("last_directory", os.path.dirname(path))
       
    dummy_parent.deleteLater()
    
    player_widget = VideoPlayerWidget(media_path=path, api_keys=api_keys)
    player_widget.show()
    
    def handle_new_connection():
        client = server.nextPendingConnection()
        if client.waitForReadyRead(1000):
            new_path = client.readAll().data().decode('utf-8')
            if new_path and os.path.exists(new_path):
                player_widget.load_video(new_path)
        
        player_widget.setWindowState(player_widget.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
        player_widget.raise_()
        player_widget.activateWindow()
        client.disconnectFromServer()
        
    server.newConnection.connect(handle_new_connection)

    player_widget.setWindowState(player_widget.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
    player_widget.raise_()
    player_widget.activateWindow()

    def wake_up_nvda():
        # NVDA sometimes stays silent on the first announcement after a window opens.
        # Flashing an invisible modal dialog open-and-closed nudges the screen reader's
        # focus tracking so the startup model announcement below is actually heard.
        phantom = QDialog(player_widget)
        phantom.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        phantom.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        phantom.setGeometry(0, 0, 0, 0)
        QTimer.singleShot(50, phantom.accept)
        phantom.exec()
        player_widget.video_widget.setFocus()
        
    QTimer.singleShot(600, wake_up_nvda)
    QTimer.singleShot(1200, player_widget.announce_startup_model)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()