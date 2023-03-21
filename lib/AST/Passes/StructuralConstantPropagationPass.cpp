#include "marco/AST/Passes/StructuralConstantPropagationPass.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  StructuralConstantPropagationPass::StructuralConstantPropagationPass(diagnostic::DiagnosticEngine& diagnostics)
    : Pass(diagnostics)
  {
  }

  template<>
  bool StructuralConstantPropagationPass::run<Class>(Class& cls)
  {
    return cls.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(cls);
    });
  }

  bool StructuralConstantPropagationPass::run(std::unique_ptr<Class>& cls)
  {
    return run<Class>(*cls);
  }

  template<>
  bool StructuralConstantPropagationPass::run<PartialDerFunction>(Class& cls)
  {
    // A partial der function doesn't have a body.
    // And even if it had one, it would not impact on the model structure.
    return true;
  }

  template<>
  bool StructuralConstantPropagationPass::run<StandardFunction>(Class& cls)
  {
    // Nothing to do. Functions do not impact on the model structure.
    return true;
  }

  template<>
  bool StructuralConstantPropagationPass::run<Model>(Class& cls)
  {
    auto* model = cls.get<Model>();
    SymbolTableScope varScope(symbolTable);

    for (auto& member : model->getMembers()) {
      symbolTable.insert(member->getName(), Symbol(*member));
    }

    for (auto& member : model->getMembers()) {
      if (!run(*member)) {
        return false;
      }
    }

    for (auto& equationsBlock : model->getEquationsBlocks()) {
      for (auto& forEquation : equationsBlock->getForEquations()) {
        if (!run(*forEquation)) {
          return false;
        }
      }
    }

    for (auto& equationsBlock : model->getInitialEquationsBlocks()) {
      for (auto& forEquation : equationsBlock->getForEquations()) {
        if (!run(*forEquation)) {
          return false;
        }
      }
    }

    return true;
  }

  template<>
  bool StructuralConstantPropagationPass::run<Package>(Class& cls)
  {
    auto* package = cls.get<Package>();

    for (auto& innerClass : *package) {
      if (!run<Class>(*innerClass))
        return false;
    }

    return true;
  }

  template<>
  bool StructuralConstantPropagationPass::run<Record>(Class& cls)
  {
    llvm_unreachable("Not implemented");
    return false;
  }

  bool StructuralConstantPropagationPass::run(ForEquation& forEquation)
  {
    for (auto& ind : forEquation.getInductions()) {
      if (!run<Expression>(*ind->getBegin())) {
        return false;
      }

      if (!ind->getBegin()->isa<Constant>()) {
        // TODO emit error through diagnostic engine
        llvm::errs() << "Can't compute begin index\n";
        return false;
      }

      if (!run<Expression>(*ind->getEnd())) {
        return false;
      }

      if (!ind->getEnd()->isa<Constant>()) {
        // TODO emit error through diagnostic engine
        llvm::errs() << "Can't compute end index\n";
        return false;
      }
    }

    return true;
  }

  bool StructuralConstantPropagationPass::run(Member& member)
  {
    auto numOfErrors = diagnostics()->numOfErrors();

    for (auto& dimension : member.getType().getDimensions()) {
      if (dimension.hasExpression()) {
        if (!run<Expression>(*dimension.getExpression())) {
          return false;
        }
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool StructuralConstantPropagationPass::run<Expression>(Expression& expression)
  {
    return expression.visit([&](auto& obj) {
      using type = decltype(obj);
      using deref = typename std::remove_reference<type>::type;
      using deconst = typename std::remove_const<deref>::type;
      return run<deconst>(expression);
    });
  }

  template<>
  bool StructuralConstantPropagationPass::run<Array>(Expression& expression)
  {
    auto* array = expression.get<Array>();

    for (auto& element : *array) {
      if (!run<Expression>(*element)) {
        return false;
      }
    }

    return true;
  }

  template<>
  bool StructuralConstantPropagationPass::run<Call>(Expression& expression)
  {
    auto* call = expression.get<Call>();

    if (!run<Expression>(*call->getFunction())) {
      return false;
    }

    for (auto& arg : *call) {
      if (!run<Expression>(*arg)) {
        return false;
      }
    }

    return true;
  }

  template<>
  bool StructuralConstantPropagationPass::run<Constant>(Expression& expression)
  {
    // Nothing to do
    return true;
  }

  template<>
  bool StructuralConstantPropagationPass::run<Operation>(Expression& expression)
  {
    auto* operation = expression.get<Operation>();

    for (auto& arg : *operation) {
      if (!run<Expression>(*arg)) {
        return false;
      }
    }

    return true;
  }

  template<>
  bool StructuralConstantPropagationPass::run<ReferenceAccess>(Expression& expression)
  {
    auto numOfErrors = diagnostics()->numOfErrors();
    auto* reference = expression.get<ReferenceAccess>();

    if (symbolTable.count(reference->getName()) == 0) {
      // Built-in variables (such as time) or functions are not in the symbol table.
      return true;
    }

    const auto& symbol = symbolTable.lookup(reference->getName());
    const auto* member = symbol.get<Member>();

    if (member->hasModification()) {
      if (auto* modification = member->getModification(); modification->hasExpression()) {
        expression = *modification->getExpression();
      }
    }

    return numOfErrors == diagnostics()->numOfErrors();
  }

  template<>
  bool StructuralConstantPropagationPass::run<Tuple>(Expression& expression)
  {
    auto* tuple = expression.get<Tuple>();

    for (auto& element : *tuple) {
      if (!run<Expression>(*element))
        return false;
    }

    return true;
  }

  std::unique_ptr<Pass> createStructuralConstantPropagationPass(diagnostic::DiagnosticEngine& diagnostics)
  {
    return std::make_unique<StructuralConstantPropagationPass>(diagnostics);
  }
}
