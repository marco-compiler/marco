#ifndef MARCO_DIALECT_RUNTIME_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define MARCO_DIALECT_RUNTIME_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace runtime {
void registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);
}
} // namespace mlir

#endif // MARCO_DIALECT_RUNTIME_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
