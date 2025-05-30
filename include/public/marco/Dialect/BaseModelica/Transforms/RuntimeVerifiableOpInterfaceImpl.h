#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RUNTIMEVERIFIABLEOPINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RUNTIMEVERIFIABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace bmodelica {
void registerRuntimeVerifiableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);
}
} // namespace mlir

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RUNTIMEVERIFIABLEOPINTERFACEIMPL_H
