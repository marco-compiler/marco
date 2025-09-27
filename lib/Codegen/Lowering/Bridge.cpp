#include "marco/Codegen/Lowering/BaseModelica/Bridge.h"
#include "marco/Codegen/Lowering/BaseModelica/AlgorithmLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/ArrayGeneratorLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/AssignmentStatementLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/BreakStatementLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/CallLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/CallStatementLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/ClassLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/ComponentReferenceLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/ConstantLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/EqualityEquationLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/EquationLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/EquationSectionLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/ExpressionLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/ForEquationLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/ForStatementLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/IfEquationLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/IfStatementLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/LoweringContext.h"
#include "marco/Codegen/Lowering/BaseModelica/ModelLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/OperationLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/PackageLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/PartialDerFunctionLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/RecordLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/ReturnStatementLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/StandardFunctionLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/StatementLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/SubscriptLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/TupleLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/WhenEquationLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/WhenStatementLowerer.h"
#include "marco/Codegen/Lowering/BaseModelica/WhileStatementLowerer.h"
#include <memory>

using namespace ::marco;
using namespace ::marco::ast::bmodelica;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
BridgeInterface::~BridgeInterface() = default;

class Bridge::Impl : public BridgeInterface {
public:
  Impl(mlir::MLIRContext &context);

  ~Impl() override;

  LoweringContext &getContext() override;

  const LoweringContext &getContext() const override;

  mlir::Operation *getRoot() const override;

  std::unique_ptr<mlir::ModuleOp> &getMLIRModule();

  [[nodiscard]] bool convert(const Root &root);

  void declare(const Class &node) override;

  void declare(const Model &node) override;

  void declare(const Package &node) override;

  void declare(const PartialDerFunction &node) override;

  void declare(const Record &node) override;

  void declare(const StandardFunction &node) override;

  [[nodiscard]] bool declareVariables(const Class &node) override;

  [[nodiscard]] bool declareVariables(const Model &model) override;

  [[nodiscard]] bool declareVariables(const Package &package) override;

  [[nodiscard]] bool
  declareVariables(const PartialDerFunction &function) override;

  [[nodiscard]] bool declareVariables(const Record &record) override;

  [[nodiscard]] bool
  declareVariables(const StandardFunction &function) override;

  [[nodiscard]] bool declare(const Member &node) override;

  [[nodiscard]] bool lower(const Class &node) override;

  [[nodiscard]] bool lower(const Model &node) override;

  [[nodiscard]] bool lower(const Package &node) override;

  [[nodiscard]] bool lower(const PartialDerFunction &node) override;

  [[nodiscard]] bool lower(const Record &node) override;

  [[nodiscard]] bool lower(const StandardFunction &node) override;

  [[nodiscard]] bool lowerClassBody(const Class &node) override;

  [[nodiscard]] bool
  createBindingEquation(const Member &variable,
                        const Expression &expression) override;

  [[nodiscard]] bool lowerStartAttribute(mlir::SymbolRefAttr variable,
                                         const Expression &expression,
                                         bool fixed, bool each) override;

  std::optional<Results> lower(const Expression &expression) override;

  std::optional<Results> lower(const ArrayGenerator &node) override;

  std::optional<Results> lower(const Call &node) override;

  std::optional<Results> lower(const Constant &constant) override;

  std::optional<Results> lower(const Operation &operation) override;

  std::optional<Results>
  lower(const ComponentReference &componentReference) override;

  std::optional<Results> lower(const Tuple &tuple) override;

  std::optional<Results> lower(const Subscript &subscript) override;

  [[nodiscard]] bool lower(const EquationSection &node) override;

  [[nodiscard]] bool lower(const Equation &node) override;

  [[nodiscard]] bool lower(const EqualityEquation &node) override;

  [[nodiscard]] bool lower(const IfEquation &node) override;

  [[nodiscard]] bool lower(const ForEquation &node) override;

  [[nodiscard]] bool lower(const WhenEquation &node) override;

  [[nodiscard]] bool lower(const Algorithm &node) override;

  [[nodiscard]] bool lower(const Statement &node) override;

  [[nodiscard]] bool lower(const AssignmentStatement &statement) override;

  [[nodiscard]] bool
  lowerAssignmentToComponentReference(mlir::Location assignmentLoc,
                                      const ComponentReference &destination,
                                      mlir::Value value) override;

  [[nodiscard]] bool lower(const BreakStatement &statement) override;

  [[nodiscard]] bool lower(const CallStatement &statement) override;

  [[nodiscard]] bool lower(const ForStatement &statement) override;

  [[nodiscard]] bool lower(const IfStatement &statement) override;

  [[nodiscard]] bool lower(const ReturnStatement &statement) override;

  [[nodiscard]] bool lower(const WhenStatement &statement) override;

  [[nodiscard]] bool lower(const WhileStatement &statement) override;

private:
  std::unique_ptr<LoweringContext> context;

  // The module that is populated while converting the AST.
  std::unique_ptr<mlir::ModuleOp> module;

  // Lowerers.
  std::unique_ptr<ClassLowerer> classLowerer;
  std::unique_ptr<ModelLowerer> modelLowerer;
  std::unique_ptr<PackageLowerer> packageLowerer;
  std::unique_ptr<RecordLowerer> recordLowerer;
  std::unique_ptr<PartialDerFunctionLowerer> partialDerFunctionLowerer;
  std::unique_ptr<StandardFunctionLowerer> standardFunctionLowerer;
  std::unique_ptr<ExpressionLowerer> expressionLowerer;
  std::unique_ptr<ArrayGeneratorLowerer> arrayLowerer;
  std::unique_ptr<CallLowerer> callLowerer;
  std::unique_ptr<ConstantLowerer> constantLowerer;
  std::unique_ptr<OperationLowerer> operationLowerer;
  std::unique_ptr<ComponentReferenceLowerer> componentReferenceLowerer;
  std::unique_ptr<TupleLowerer> tupleLowerer;
  std::unique_ptr<SubscriptLowerer> subscriptLowerer;
  std::unique_ptr<EquationSectionLowerer> equationSectionLowerer;
  std::unique_ptr<EquationLowerer> equationLowerer;
  std::unique_ptr<EqualityEquationLowerer> equalityEquationLowerer;
  std::unique_ptr<ForEquationLowerer> forEquationLowerer;
  std::unique_ptr<IfEquationLowerer> ifEquationLowerer;
  std::unique_ptr<WhenEquationLowerer> whenEquationLowerer;
  std::unique_ptr<AlgorithmLowerer> algorithmLowerer;
  std::unique_ptr<StatementLowerer> statementLowerer;
  std::unique_ptr<AssignmentStatementLowerer> assignmentStatementLowerer;
  std::unique_ptr<BreakStatementLowerer> breakStatementLowerer;
  std::unique_ptr<CallStatementLowerer> callStatementLowerer;
  std::unique_ptr<ForStatementLowerer> forStatementLowerer;
  std::unique_ptr<IfStatementLowerer> ifStatementLowerer;
  std::unique_ptr<ReturnStatementLowerer> returnStatementLowerer;
  std::unique_ptr<WhenStatementLowerer> whenStatementLowerer;
  std::unique_ptr<WhileStatementLowerer> whileStatementLowerer;
};

Bridge::Impl::Impl(mlir::MLIRContext &context) {
  this->context = std::make_unique<LoweringContext>(context);

  this->module = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(this->context->getBuilder().getUnknownLoc()));

  // Initialize the lowerers.
  this->classLowerer = std::make_unique<ClassLowerer>(this);
  this->modelLowerer = std::make_unique<ModelLowerer>(this);
  this->packageLowerer = std::make_unique<PackageLowerer>(this);

  this->partialDerFunctionLowerer =
      std::make_unique<PartialDerFunctionLowerer>(this);

  this->recordLowerer = std::make_unique<RecordLowerer>(this);

  this->standardFunctionLowerer =
      std::make_unique<StandardFunctionLowerer>(this);

  this->expressionLowerer = std::make_unique<ExpressionLowerer>(this);
  this->arrayLowerer = std::make_unique<ArrayGeneratorLowerer>(this);
  this->callLowerer = std::make_unique<CallLowerer>(this);

  this->componentReferenceLowerer =
      std::make_unique<ComponentReferenceLowerer>(this);

  this->constantLowerer = std::make_unique<ConstantLowerer>(this);
  this->operationLowerer = std::make_unique<OperationLowerer>(this);
  this->tupleLowerer = std::make_unique<TupleLowerer>(this);
  this->subscriptLowerer = std::make_unique<SubscriptLowerer>(this);

  this->equationSectionLowerer = std::make_unique<EquationSectionLowerer>(this);

  this->equationLowerer = std::make_unique<EquationLowerer>(this);

  this->equalityEquationLowerer =
      std::make_unique<EqualityEquationLowerer>(this);

  this->forEquationLowerer = std::make_unique<ForEquationLowerer>(this);
  this->ifEquationLowerer = std::make_unique<IfEquationLowerer>(this);
  this->whenEquationLowerer = std::make_unique<WhenEquationLowerer>(this);

  this->algorithmLowerer = std::make_unique<AlgorithmLowerer>(this);
  this->statementLowerer = std::make_unique<StatementLowerer>(this);

  this->assignmentStatementLowerer =
      std::make_unique<AssignmentStatementLowerer>(this);

  this->breakStatementLowerer = std::make_unique<BreakStatementLowerer>(this);
  this->callStatementLowerer = std::make_unique<CallStatementLowerer>(this);
  this->forStatementLowerer = std::make_unique<ForStatementLowerer>(this);
  this->ifStatementLowerer = std::make_unique<IfStatementLowerer>(this);

  this->returnStatementLowerer = std::make_unique<ReturnStatementLowerer>(this);

  this->whenStatementLowerer = std::make_unique<WhenStatementLowerer>(this);

  this->whileStatementLowerer = std::make_unique<WhileStatementLowerer>(this);
}

Bridge::Impl::~Impl() {
  if (module != nullptr) {
    module->erase();
  }
}

LoweringContext &Bridge::Impl::getContext() {
  assert(context != nullptr && "Lowering context not set");
  return *context;
}

const LoweringContext &Bridge::Impl::getContext() const {
  assert(context != nullptr && "Lowering context not set");
  return *context;
}

mlir::Operation *Bridge::Impl::getRoot() const {
  assert(module != nullptr && "MLIR module not created");
  return module->getOperation();
}

bool Bridge::Impl::convert(const Root &root) {
  mlir::OpBuilder::InsertionGuard guard(context->getBuilder());
  context->pushLookupScope(module->getOperation());
  context->getBuilder().setInsertionPointToStart(module->getBody());

  for (const auto &cls : root.getInnerClasses()) {
    classLowerer->declare(*cls->cast<Class>());
  }

  for (const auto &cls : root.getInnerClasses()) {
    if (!classLowerer->declareVariables(*cls->cast<Class>())) {
      return false;
    }
  }

  for (const auto &cls : root.getInnerClasses()) {
    if (!classLowerer->lower(*cls->cast<Class>())) {
      return false;
    }
  }

  return true;
}

std::unique_ptr<mlir::ModuleOp> &Bridge::Impl::getMLIRModule() {
  assert(module != nullptr);
  return module;
}

void Bridge::Impl::declare(const Class &cls) {
  assert(classLowerer != nullptr);
  return classLowerer->declare(cls);
}

void Bridge::Impl::declare(const Model &model) {
  assert(modelLowerer != nullptr);
  return modelLowerer->declare(model);
}

void Bridge::Impl::declare(const Package &package) {
  assert(packageLowerer != nullptr);
  return packageLowerer->declare(package);
}

void Bridge::Impl::declare(const PartialDerFunction &function) {
  assert(partialDerFunctionLowerer != nullptr);
  return partialDerFunctionLowerer->declare(function);
}

void Bridge::Impl::declare(const Record &record) {
  assert(recordLowerer != nullptr);
  return recordLowerer->declare(record);
}

void Bridge::Impl::declare(const StandardFunction &function) {
  assert(standardFunctionLowerer != nullptr);
  return standardFunctionLowerer->declare(function);
}

bool Bridge::Impl::declareVariables(const Class &cls) {
  assert(classLowerer != nullptr);
  return classLowerer->declareVariables(cls);
}

bool Bridge::Impl::declareVariables(const Model &model) {
  assert(modelLowerer != nullptr);
  return modelLowerer->declareVariables(model);
}

bool Bridge::Impl::declareVariables(const Package &package) {
  assert(packageLowerer != nullptr);
  return packageLowerer->declareVariables(package);
}

bool Bridge::Impl::declareVariables(const PartialDerFunction &function) {
  assert(partialDerFunctionLowerer != nullptr);
  return partialDerFunctionLowerer->declareVariables(function);
}

bool Bridge::Impl::declareVariables(const Record &record) {
  assert(recordLowerer != nullptr);
  return recordLowerer->declareVariables(record);
}

bool Bridge::Impl::declareVariables(const StandardFunction &function) {
  assert(standardFunctionLowerer != nullptr);
  return standardFunctionLowerer->declareVariables(function);
}

bool Bridge::Impl::declare(const Member &variable) {
  assert(classLowerer != nullptr);
  return classLowerer->declare(variable);
}

bool Bridge::Impl::lower(const Class &cls) {
  assert(classLowerer != nullptr);
  return classLowerer->lower(cls);
}

bool Bridge::Impl::lower(const Model &model) {
  assert(modelLowerer != nullptr);
  return modelLowerer->lower(model);
}

bool Bridge::Impl::lower(const Package &package) {
  assert(packageLowerer != nullptr);
  return packageLowerer->lower(package);
}

bool Bridge::Impl::lower(const PartialDerFunction &function) {
  assert(partialDerFunctionLowerer != nullptr);
  return partialDerFunctionLowerer->lower(function);
}

bool Bridge::Impl::lower(const Record &record) {
  assert(recordLowerer != nullptr);
  return recordLowerer->lower(record);
}

bool Bridge::Impl::lower(const StandardFunction &function) {
  assert(standardFunctionLowerer != nullptr);
  return standardFunctionLowerer->lower(function);
}

bool Bridge::Impl::lowerClassBody(const Class &cls) {
  assert(classLowerer != nullptr);
  return classLowerer->lowerClassBody(cls);
}

bool Bridge::Impl::createBindingEquation(const Member &variable,
                                         const Expression &expression) {
  assert(classLowerer != nullptr);
  return classLowerer->createBindingEquation(variable, expression);
}

bool Bridge::Impl::lowerStartAttribute(mlir::SymbolRefAttr variable,
                                       const Expression &expression, bool fixed,
                                       bool each) {
  assert(classLowerer != nullptr);

  return classLowerer->lowerStartAttribute(variable, expression, fixed, each);
}

std::optional<Results> Bridge::Impl::lower(const Expression &expression) {
  assert(expressionLowerer != nullptr);
  return expressionLowerer->lower(expression);
}

std::optional<Results> Bridge::Impl::lower(const ArrayGenerator &array) {
  assert(arrayLowerer != nullptr);
  return arrayLowerer->lower(array);
}

std::optional<Results> Bridge::Impl::lower(const Call &call) {
  assert(callLowerer != nullptr);
  return callLowerer->lower(call);
}

std::optional<Results> Bridge::Impl::lower(const Constant &constant) {
  assert(constantLowerer != nullptr);
  return constantLowerer->lower(constant);
}

std::optional<Results> Bridge::Impl::lower(const Operation &operation) {
  assert(operationLowerer != nullptr);
  return operationLowerer->lower(operation);
}

std::optional<Results>
Bridge::Impl::lower(const ComponentReference &componentReference) {
  assert(componentReferenceLowerer != nullptr);
  return componentReferenceLowerer->lower(componentReference);
}

std::optional<Results> Bridge::Impl::lower(const Tuple &tuple) {
  assert(tupleLowerer != nullptr);
  return tupleLowerer->lower(tuple);
}

std::optional<Results> Bridge::Impl::lower(const Subscript &subscript) {
  assert(subscriptLowerer != nullptr);
  return subscriptLowerer->lower(subscript);
}

bool Bridge::Impl::lower(const EquationSection &equationSection) {
  assert(equationLowerer != nullptr);
  return equationSectionLowerer->lower(equationSection);
}

bool Bridge::Impl::lower(const Equation &equation) {
  assert(equationLowerer != nullptr);
  return equationLowerer->lower(equation);
}

bool Bridge::Impl::lower(const EqualityEquation &equation) {
  assert(equationLowerer != nullptr);
  return equalityEquationLowerer->lower(equation);
}

bool Bridge::Impl::lower(const ForEquation &forEquation) {
  assert(equationLowerer != nullptr);
  return forEquationLowerer->lower(forEquation);
}

bool Bridge::Impl::lower(const IfEquation &equation) {
  assert(equationLowerer != nullptr);
  return ifEquationLowerer->lower(equation);
}

bool Bridge::Impl::lower(const WhenEquation &equation) {
  assert(equationLowerer != nullptr);
  return whenEquationLowerer->lower(equation);
}

bool Bridge::Impl::lower(const Algorithm &algorithm) {
  assert(algorithmLowerer != nullptr);
  return algorithmLowerer->lower(algorithm);
}

bool Bridge::Impl::lower(const Statement &statement) {
  assert(statementLowerer != nullptr);
  return statementLowerer->lower(statement);
}

bool Bridge::Impl::lower(const AssignmentStatement &statement) {
  assert(assignmentStatementLowerer != nullptr);
  return assignmentStatementLowerer->lower(statement);
}

[[nodiscard]] bool Bridge::Impl::lowerAssignmentToComponentReference(
    mlir::Location assignmentLoc, const ComponentReference &destination,
    mlir::Value value) {
  assert(assignmentStatementLowerer != nullptr);
  return assignmentStatementLowerer->lowerAssignmentToComponentReference(
      assignmentLoc, destination, value);
}

bool Bridge::Impl::lower(const BreakStatement &statement) {
  assert(breakStatementLowerer != nullptr);
  return breakStatementLowerer->lower(statement);
}

bool Bridge::Impl::lower(const CallStatement &statement) {
  assert(callStatementLowerer != nullptr);
  return callStatementLowerer->lower(statement);
}

bool Bridge::Impl::lower(const ForStatement &statement) {
  assert(forStatementLowerer != nullptr);
  return forStatementLowerer->lower(statement);
}

bool Bridge::Impl::lower(const IfStatement &statement) {
  assert(ifStatementLowerer != nullptr);
  return ifStatementLowerer->lower(statement);
}

bool Bridge::Impl::lower(const ReturnStatement &statement) {
  assert(returnStatementLowerer != nullptr);
  return returnStatementLowerer->lower(statement);
}

bool Bridge::Impl::lower(const WhenStatement &statement) {
  assert(whenStatementLowerer != nullptr);
  return whenStatementLowerer->lower(statement);
}

bool Bridge::Impl::lower(const WhileStatement &statement) {
  assert(whileStatementLowerer != nullptr);
  return whileStatementLowerer->lower(statement);
}

Bridge::Bridge(mlir::MLIRContext &context)
    : impl(std::make_unique<Impl>(context)) {}

Bridge::~Bridge() = default;

bool Bridge::lower(const Root &root) { return impl->convert(root); }

std::unique_ptr<mlir::ModuleOp> &Bridge::getMLIRModule() {
  return impl->getMLIRModule();
}
} // namespace marco::codegen::lowering::bmodelica
