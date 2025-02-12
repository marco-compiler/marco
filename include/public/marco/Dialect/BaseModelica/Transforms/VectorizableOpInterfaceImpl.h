#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_VECTORIZABLEOPINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_VECTORIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace bmodelica {
void registerVectorizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);
}
} // namespace mlir

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_VECTORIZABLEOPINTERFACEIMPL_H
