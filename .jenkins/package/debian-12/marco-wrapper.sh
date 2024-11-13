#!/bin/sh

# Run the driver with the modified arguments.
/usr/lib/marco/marco "$@" -L /usr/lib/marco-runtime -Wl,-rpath /usr/lib/marco-runtime
