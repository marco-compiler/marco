#ifndef MARCO_DIALECT_RUNTIME_TRANSFORMS_ALLINTERFACES_H
#define MARCO_DIALECT_RUNTIME_TRANSFORMS_ALLINTERFACES_H

namespace mlir {
class DialectRegistry;

namespace runtime {
void registerAllDialectInterfaceImplementations(
    mlir::DialectRegistry &registry);
}
} // namespace mlir

#endif // MARCO_DIALECT_RUNTIME_TRANSFORMS_ALLINTERFACES_H
