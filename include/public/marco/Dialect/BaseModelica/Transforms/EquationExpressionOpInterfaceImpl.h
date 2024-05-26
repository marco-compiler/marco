#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONEXPRESSIONOPINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONEXPRESSIONOPINTERFACEIMPL_H

namespace mlir
{
  class DialectRegistry;

  namespace bmodelica
  {
    void registerEquationExpressionOpInterfaceExternalModels(
        mlir::DialectRegistry& registry);
  }
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONEXPRESSIONOPINTERFACEIMPL_H
