#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_ALLOCATIONOPINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_ALLOCATIONOPINTERFACEIMPL_H

namespace mlir
{
  class DialectRegistry;

  namespace bmodelica
  {
    void registerAllocationOpInterfaceExternalModels(
        mlir::DialectRegistry& registry);
  }
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_ALLOCATIONOPINTERFACEIMPL_H
