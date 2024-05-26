#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_ALLINTERFACES_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_ALLINTERFACES_H

namespace mlir
{
  class DialectRegistry;

  namespace bmodelica
  {
    void registerAllDialectInterfaceImplementations(
        mlir::DialectRegistry& registry);
  }
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_ALLINTERFACES_H
