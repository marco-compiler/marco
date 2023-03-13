#ifndef MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONBRIDGE_H
#define MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONBRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class StandardFunctionLowerer : public Lowerer
  {
    public:
      StandardFunctionLowerer(LoweringContext* context, BridgeInterface* bridge);

      std::vector<mlir::Operation*> lower(const ast::StandardFunction& function);

    protected:
      using Lowerer::lower;

    private:
      /// Lower a member of a function.
      ///
      /// Input members are ignored because they are supposed to be unmodifiable
      /// as per the Modelica standard, and thus don't need a local copy.
      /// Output arrays are always allocated on the heap and eventually moved to
      /// input arguments by the dedicated pass. Protected arrays, instead, are
      /// allocated according to the ArrayType allocation logic.
      void lower(const ast::Member& member);

      void lowerVariableDefaultValue(
          llvm::StringRef variable, const ast::Expression& expression);
  };
}

#endif // MARCO_CODEGEN_LOWERING_STANDARDFUNCTIONBRIDGE_H
