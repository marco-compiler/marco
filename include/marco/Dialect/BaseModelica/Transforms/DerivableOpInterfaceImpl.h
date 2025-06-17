#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DERIVABLEOPINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DERIVABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace bmodelica {
void registerDerivableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);
}
} // namespace mlir

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DERIVABLEOPINTERFACEIMPL_H
