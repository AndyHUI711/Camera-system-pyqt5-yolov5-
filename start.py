##Copyright
##written by HUI, CHEUNG YUEN
##Student of HKUST
##FYP, FINAL YEAR PROJECT
from PyQt5 import QtWidgets

from controller import MainWindow_controller
from lib.share import shareInfo


"""""""""""
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    shareInfo.loginWin = MainWindow_controller()
    shareInfo.loginWin.show()
    sys.exit(app.exec_())
    
    self.ui_yolo.textBrowser.setText(s+": "+label)
    
        if self.stop_yolo == True:
            self.timer_camera_y.stop()
            self.ui_yolo.textBrowser.clear()
            self.ui_yolo.label_3.clear()
            return
"""""""""""
#pip install PyQt5
#pip install PyQt5-tools
#pyuic5 -x UI.ui -o UI.py
#pip install opencv-contrib-python
#pip install opencv-python
