#pragma once

// Just a convenience header file to include all the frontend passes and the
// pass manager to run them.

#include "passes/ConstantFoldingPass.h"
#include "passes/TypeCheckingPass.h"
#include "PassManager.h"
