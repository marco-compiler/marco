#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INVERTIBLEOPINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INVERTIBLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace bmodelica {
void registerInvertibleOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);
}
} // namespace mlir

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_INVERTIBLEOPINTERFACEIMPL_H
