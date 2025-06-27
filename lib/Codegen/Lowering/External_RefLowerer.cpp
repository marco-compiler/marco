#include "marco/Codegen/Lowering/External_RefLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
  External_RefLowerer::External_RefLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

  bool External_RefLowerer::lower(const ast::ExternalRef &er) {
    if (er.empty()) {
      return true;
    }

    builder().create<mlir::bmodelica::CallOp>(
      loc(er.getLocation()),
      llvm::SmallVector<mlir::Type>{},
      builder().getSymbolRefAttr(er->getExternalFunctionCall()->getName()),
    llvm::SmallVector<mlir::Value>{} // No arguments
    );
    
    return true;
  } // namespace marco::codegen::lowering
}

