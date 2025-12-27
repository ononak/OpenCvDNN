#include <QtWidgets/QApplication>
#include "OpenCvDnnWidget.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    OpenCvDNNWidget window;
    window.show();

    return app.exec();
}
