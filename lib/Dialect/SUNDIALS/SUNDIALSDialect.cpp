#include "marco/Dialect/SUNDIALS/Attributes.h"
#include "marco/Dialect/SUNDIALS/SUNDIALSDialect.h"
#include "marco/Dialect/SUNDIALS/Ops.h"
#include "marco/Dialect/SUNDIALS/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::mlir::sundials;

#include "marco/Dialect/SUNDIALS/SUNDIALSDialect.cpp.inc"

//===---------------------------------------------------------------------===//
// SUNDIALS dialect
//===---------------------------------------------------------------------===//

void SUNDIALSDialect::initialize() {
  addOperations<
      #define GET_OP_LIST
      #include "marco/Dialect/SUNDIALS/SUNDIALS.cpp.inc"
  >();

  addTypes<
      #define GET_TYPEDEF_LIST
      #include "marco/Dialect/SUNDIALS/SUNDIALSTypes.cpp.inc"
  >();

  addAttributes<
      #define GET_ATTRDEF_LIST
      #include "marco/Dialect/SUNDIALS/SUNDIALSAttributes.cpp.inc"
  >();
}

//===---------------------------------------------------------------------===//
// Tablegen type definitions
//===---------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/SUNDIALS/SUNDIALSTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/SUNDIALS/SUNDIALSAttributes.cpp.inc"
