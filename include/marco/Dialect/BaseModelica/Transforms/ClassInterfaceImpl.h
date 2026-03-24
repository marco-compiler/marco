#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_CLASSINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_CLASSINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace bmodelica {
void registerClassInterfaceExternalModels(mlir::DialectRegistry &registry);
}
} // namespace mlir

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_CLASSINTERFACEIMPL_H
