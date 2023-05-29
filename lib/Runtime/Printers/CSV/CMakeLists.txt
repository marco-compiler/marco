set(LIBS
    PUBLIC argh)

marco_add_runtime_library(runtimePrinterCSV
    STATIC
    ../Printer.cpp
    CLI.cpp
    Options.cpp
    Printer.cpp
    Profiler.cpp

    LINK_LIBS
    ${LIBS})
