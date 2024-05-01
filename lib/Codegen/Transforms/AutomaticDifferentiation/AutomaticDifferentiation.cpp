#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Dialect/BaseModelica/ModelicaDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_AUTOMATICDIFFERENTIATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  template<class T>
  unsigned int numDigits(T number)
  {
    unsigned int digits = 0;

    while (number != 0) {
      number /= 10;
      ++digits;
    }

    return digits;
  }
}

namespace mlir::bmodelica
{
  std::string getFullDerVariableName(
      llvm::StringRef baseName, unsigned int order)
  {
    assert(order > 0);

    if (order == 1) {
      return "der_" + baseName.str();
    }

    return "der_" + std::to_string(order) + "_" + baseName.str();
  }

  std::string getNextFullDerVariableName(
      llvm::StringRef currentName, unsigned int requestedOrder)
  {
    if (requestedOrder == 1) {
      return getFullDerVariableName(currentName, requestedOrder);
    }

    assert(currentName.rfind("der_") == 0);

    if (requestedOrder == 2) {
      return getFullDerVariableName(currentName.substr(4), requestedOrder);
    }

    return getFullDerVariableName(
        currentName.substr(5 + numDigits(requestedOrder - 1)),
        requestedOrder);
  }

  void mapFullDerivatives(
      mlir::Operation* classOp,
      mlir::SymbolTableCollection& symbolTableCollection,
      llvm::DenseMap<mlir::StringAttr, mlir::StringAttr>& mapping)
  {
    mlir::SymbolTable& symbolTable =
        symbolTableCollection.getSymbolTable(classOp);

    // TODO leverage OneRegion trait when available for ClassInterface.
    for (VariableOp variableOp : classOp->getRegion(0).getOps<VariableOp>()) {
      // Given a variable "x", first search for "der_x". If it doesn't exist,
      // then also "der_2_x", "der_3_x", etc. will not exist, and thus we can
      // say that "x" has no derivatives. If it exists, add the first order
      // derivative and then search for the higher order ones.

      std::string candidateFirstOrderDer =
          getFullDerVariableName(variableOp.getSymName(), 1);

      auto derivativeVariableOp =
          symbolTable.lookup<VariableOp>(candidateFirstOrderDer);

      if (!derivativeVariableOp) {
        continue;
      }

      mapping[variableOp.getSymNameAttr()] =
          derivativeVariableOp.getSymNameAttr();

      unsigned int order = 2;
      bool found;

      do {
        std::string nextName =
            getFullDerVariableName(variableOp.getSymName(), order);

        auto nextDerivativeVariableOp =
            symbolTable.lookup<VariableOp>(nextName);

        found = nextDerivativeVariableOp != nullptr;

        if (found) {
          mapping[derivativeVariableOp.getSymNameAttr()] =
              nextDerivativeVariableOp.getSymNameAttr();

          derivativeVariableOp = nextDerivativeVariableOp;
          ++order;
        }
      } while (found);
    }
  }
}

namespace
{
  class AutomaticDifferentiationPass
      : public impl::AutomaticDifferentiationPassBase<
          AutomaticDifferentiationPass>
  {
    public:
      using AutomaticDifferentiationPassBase::AutomaticDifferentiationPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(createFullDerFunctions())) {
          mlir::emitError(
              getOperation().getLoc(),
              "Error in creating the functions full derivatives");

          return signalPassFailure();
        }

        if (mlir::failed(createPartialDerFunctions())) {
          mlir::emitError(
              getOperation().getLoc(),
              "Error in creating the functions partial derivatives");

          return signalPassFailure();
        }

        if (mlir::failed(resolveTrivialDerCalls())) {
          mlir::emitError(
              getOperation().getLoc(),
              "Error in resolving the trivial derivative calls");

          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult createFullDerFunctions()
      {
        auto module = getOperation();
        mlir::OpBuilder builder(module);

        llvm::SmallVector<FunctionOp, 3> toBeDerived;

        module->walk([&](FunctionOp op) {
          if (op->hasAttrOfType<DerivativeAttr>("derivative")) {
            toBeDerived.push_back(op);
          }
        });

        // Sort the functions so that a function derivative is computed only
        // when the base function already has its body determined.

        llvm::sort(toBeDerived, [](FunctionOp first, FunctionOp second) {
          auto annotation = first->getAttrOfType<DerivativeAttr>("derivative");
          return annotation.getName() == second.getSymName();
        });

        ForwardAD forwardAD;
        mlir::SymbolTableCollection symbolTableCollection;

        for (auto& function : toBeDerived) {
          if (mlir::failed(forwardAD.createFullDerFunction(
                  builder, function, symbolTableCollection))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createPartialDerFunctions()
      {
        auto module = getOperation();
        mlir::OpBuilder builder(module);

        llvm::SmallVector<DerFunctionOp> toBeProcessed;

        // The conversion is done in an iterative way, because new derivative
        // functions may be created while converting the existing one (i.e.
        // when a function to be derived contains a call to another function).

        auto findDerFunctions = [&]() -> bool {
          module->walk([&](DerFunctionOp op) {
            toBeProcessed.push_back(op);
          });

          return !toBeProcessed.empty();
        };

        ForwardAD forwardAD;
        mlir::SymbolTableCollection symbolTableCollection;

        while (findDerFunctions()) {
          // Sort the functions so that a function derivative is computed only
          // when the base function already has its body determined.

          llvm::sort(
              toBeProcessed,
              [](DerFunctionOp first, DerFunctionOp second) {
                return first.getSymName() == second.getDerivedFunction();
              });

          for (DerFunctionOp function : toBeProcessed) {
            if (mlir::failed(forwardAD.convertPartialDerFunction(
                    builder, function, symbolTableCollection))) {
              return mlir::failure();
            }
          }

          toBeProcessed.clear();
        }

        return mlir::success();
      }

      mlir::LogicalResult resolveTrivialDerCalls()
      {
        auto module = getOperation();
        mlir::OpBuilder builder(module);

        std::vector<DerOp> ops;

        module.walk([&](DerOp op) {
          ops.push_back(op);
        });

        ForwardAD forwardAD;
        mlir::SymbolTableCollection symbolTableCollection;

        llvm::DenseMap<
            mlir::Operation*,
            llvm::DenseMap<mlir::StringAttr, mlir::StringAttr>> symbolDerivatives;

        for (auto derOp : ops) {
          mlir::Value operand = derOp.getOperand();
          mlir::Operation* definingOp = operand.getDefiningOp();

          if (definingOp == nullptr) {
            continue;
          }

          if (auto derivableOp = mlir::dyn_cast<DerivableOpInterface>(definingOp)) {
            auto classOp = derOp->getParentOfType<ClassInterface>();

            if (classOp == nullptr) {
              continue;
            }

            mlir::IRMapping ssaDerivatives;

            if (auto it = symbolDerivatives.find(classOp.getOperation());
                it == symbolDerivatives.end()) {
              mapFullDerivatives(
                  classOp.getOperation(),
                  symbolTableCollection,
                  symbolDerivatives[classOp.getOperation()]);
            }

            llvm::SmallVector<mlir::Value> ders;

            if (mlir::failed(forwardAD.deriveTree(
                    ders, builder, derivableOp,
                    symbolDerivatives[classOp.getOperation()],
                    ssaDerivatives))) {
              continue;
            }

            if (ders.size() != derOp->getNumResults()) {
              continue;
            }

            derOp->replaceAllUsesWith(ders);
            derOp.erase();
          }
        }

        return mlir::success();
      }
  };
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass()
  {
    return std::make_unique<AutomaticDifferentiationPass>();
  }
}
