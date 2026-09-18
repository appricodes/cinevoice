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

from PySide6.QtWidgets import QApplication, QFileDialog, QWidget, QDialog
from PySide6.QtCore import QSettings, QCoreApplication, QTimer, Qt
from PySide6.QtNetwork import QLocalSocket, QLocalServer
from auth import load_api_keys, save_api_keys
from config import MEDIA_FILE_DIALOG_FILTER
from ui_components import ApiKeyDialog
from main_window import VideoPlayerWidget

IPC_SERVER_NAME = "Cinevoice_SingleInstance_Server"
# Sent when a second launch carries no file at all: it only asks the running
# window to come forward, and gives the server something to read right away
# instead of sitting out the readyRead timeout.
IPC_RAISE_TOKEN = "__cinevoice_raise__"

def bring_to_front(widget):
    widget.setWindowState(widget.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
    widget.raise_()
    widget.activateWindow()

def main():
    if hasattr(sys, '_MEIPASS'):
        plugin_path = os.path.join(sys._MEIPASS, 'PySide6', 'plugins')
        QCoreApplication.addLibraryPath(plugin_path)

    app = QApplication(sys.argv)
    app.setOrganizationName("amiri.moalla@gmail.com")
    app.setApplicationName("Cinevoice")
    
    # Another instance is already running: hand it the requested file (if any) and exit,
    # so opening a video always reuses the same window instead of starting a new one.
    socket = QLocalSocket()
    socket.connectToServer(IPC_SERVER_NAME)
    if socket.waitForConnected(500):
        socket.write((video_arg or IPC_RAISE_TOKEN).encode('utf-8'))
        socket.flush()
        socket.waitForBytesWritten(1000)
        # Disconnect politely rather than letting interpreter shutdown tear the pipe
        # down, so the running instance is sure to see the path we just sent.
        socket.disconnectFromServer()
        if socket.state() != QLocalSocket.LocalSocketState.UnconnectedState:
            socket.waitForDisconnected(1000)
        sys.exit(0)
    
    server = QLocalServer()
    server.removeServer(IPC_SERVER_NAME)
    if not server.listen(IPC_SERVER_NAME):
        pass

    # Start handling incoming paths right now, not after the window exists: the API key
    # dialog and the file picker below can keep us here indefinitely, and a path arriving
    # meanwhile would otherwise be dropped with the sending instance already gone.
    state = {"player": None, "startup_dialog": None, "pending": []}

    def handle_new_connection():
        client = server.nextPendingConnection()
        if client is None:
            return

        new_path = ""
        if client.waitForReadyRead(1000):
            new_path = client.readAll().data().decode('utf-8', 'ignore').strip()
        client.disconnectFromServer()

        if new_path and new_path != IPC_RAISE_TOKEN and os.path.exists(new_path):
            if state["player"] is not None:
                state["player"].load_video(new_path)
            else:
                # Still in the startup dialogs: remember the file, and close the file
                # picker so the video the user just opened wins over the prompt.
                state["pending"].append(new_path)
                if state["startup_dialog"] is not None:
                    state["startup_dialog"].reject()

        if state["player"] is not None:
            bring_to_front(state["player"])

    server.newConnection.connect(handle_new_connection)

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
    if not path and state["pending"]:
        path = state["pending"].pop()

    if not path:
       last_dir = q_settings.value("last_directory", ".")
       # Built as an object rather than via QFileDialog.getOpenFileName so a file
       # arriving from a second launch can close it; it stays the native dialog.
       file_dialog = QFileDialog(dummy_parent, "Select video or image file", last_dir, MEDIA_FILE_DIALOG_FILTER)
       file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
       state["startup_dialog"] = file_dialog
       accepted = file_dialog.exec() == QDialog.DialogCode.Accepted
       state["startup_dialog"] = None

       if state["pending"]:
          path = state["pending"].pop()
       elif accepted and file_dialog.selectedFiles():
          path = file_dialog.selectedFiles()[0]

       if not path:
          return
       q_settings.setValue("last_directory", os.path.dirname(path))
       
    dummy_parent.deleteLater()
    
    player_widget = VideoPlayerWidget(media_path=path, api_keys=api_keys)
    state["player"] = player_widget
    player_widget.show()

    bring_to_front(player_widget)

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
