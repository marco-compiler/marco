#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir
{
  class DialectRegistry;

  namespace bmodelica
  {
    void registerBufferizableOpInterfaceExternalModels(
        mlir::DialectRegistry& registry);
  }
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
