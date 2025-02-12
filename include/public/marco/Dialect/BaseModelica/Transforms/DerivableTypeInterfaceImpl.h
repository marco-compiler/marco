#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DERIVABLETYPEINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DERIVABLETYPEINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace bmodelica {
void registerDerivableTypeInterfaceExternalModels(
    mlir::DialectRegistry &registry);
}
} // namespace mlir

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_DERIVABLETYPEINTERFACEIMPL_H
