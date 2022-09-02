#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation/ForwardAD.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_AUTOMATICDIFFERENTIATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

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

namespace marco::codegen
{
  std::string getFullDerVariableName(llvm::StringRef baseName, unsigned int order)
  {
    assert(order > 0);

    if (order == 1) {
      return "der_" + baseName.str();
    }

    return "der_" + std::to_string(order) + "_" + baseName.str();
  }

  std::string getNextFullDerVariableName(llvm::StringRef currentName, unsigned int requestedOrder)
  {
    if (requestedOrder == 1) {
      return getFullDerVariableName(currentName, requestedOrder);
    }

    assert(currentName.rfind("der_") == 0);

    if (requestedOrder == 2) {
      return getFullDerVariableName(currentName.substr(4), requestedOrder);
    }

    return getFullDerVariableName(currentName.substr(5 + numDigits(requestedOrder - 1)), requestedOrder);
  }

  void mapFullDerivatives(mlir::BlockAndValueMapping& mapping, llvm::ArrayRef<mlir::Value> members)
  {
    llvm::StringMap<mlir::Value> membersByName;

    for (const auto& member : members) {
      auto memberOp = member.getDefiningOp<MemberCreateOp>();
      membersByName[memberOp.getSymName()] = member;
    }

    for (const auto& member : members) {
      auto name = member.getDefiningOp<MemberCreateOp>().getSymName();

      // Given a variable "x", first search for "der_x". If it doesn't exist,
      // then also "der_2_x", "der_3_x", etc. will not exist and thus we can
      // say that "x" has no derivatives. If it exists, add the first order
      // derivative and then search for the higher order ones.

      auto candidateFirstOrderDer = getFullDerVariableName(name, 1);
      auto derIt = membersByName.find(candidateFirstOrderDer);

      if (derIt == membersByName.end()) {
        continue;
      }

      mlir::Value der = derIt->second;
      mapping.map(member, der);

      unsigned int order = 2;
      bool found;

      do {
        auto nextName = getFullDerVariableName(name, order);
        auto nextDerIt = membersByName.find(nextName);
        found = nextDerIt != membersByName.end();

        if (found) {
          mlir::Value nextDer = nextDerIt->second;
          mapping.map(der, nextDer);
          der = nextDer;
        }

        ++order;
      } while (found);
    }
  }
}

namespace
{
  class AutomaticDifferentiationPass : public impl::AutomaticDifferentiationPassBase<AutomaticDifferentiationPass>
  {
    public:
      using AutomaticDifferentiationPassBase::AutomaticDifferentiationPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(createFullDerFunctions())) {
          mlir::emitError(getOperation().getLoc(), "Error in creating the functions full derivatives");
          return signalPassFailure();
        }

        if (mlir::failed(createPartialDerFunctions())) {
          mlir::emitError(getOperation().getLoc(), "Error in creating the functions partial derivatives");
          return signalPassFailure();
        }

        if (mlir::failed(resolveTrivialDerCalls())) {
          mlir::emitError(getOperation().getLoc(), "Error in resolving the trivial derivative calls");
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

        for (auto& function : toBeDerived) {
          if (auto res = forwardAD.createFullDerFunction(builder, function); mlir::failed(res)) {
            return res;
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createPartialDerFunctions()
      {
        auto module = getOperation();
        mlir::OpBuilder builder(module);

        llvm::SmallVector<DerFunctionOp, 3> toBeProcessed;

        // The conversion is done in an iterative way, because new derivative
        // functions may be created while converting the existing one (i.e. when
        // a function to be derived contains a call to an another function).

        auto findDerFunctions = [&]() -> bool {
          module->walk([&](DerFunctionOp op) {
            toBeProcessed.push_back(op);
          });

          return !toBeProcessed.empty();
        };

        ForwardAD forwardAD;

        while (findDerFunctions()) {
          // Sort the functions so that a function derivative is computed only
          // when the base function already has its body determined.

          llvm::sort(toBeProcessed, [](DerFunctionOp first, DerFunctionOp second) {
            return first.getSymName() == second.getDerivedFunction();
          });

          for (auto& function : toBeProcessed) {
            if (auto res = forwardAD.createPartialDerFunction(builder, function); mlir::failed(res)) {
              return res;
            }

            function->erase();
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

            mlir::BlockAndValueMapping derivatives;
            mapFullDerivatives(derivatives, classOp.getMembers());

            mlir::ValueRange ders = forwardAD.deriveTree(builder, derivableOp, derivatives);

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

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass()
  {
    return std::make_unique<AutomaticDifferentiationPass>();
  }
}
