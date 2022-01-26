
/*
using namespace marco::codegen;
using namespace marco::codegen::modelica;

static mlir::Value getWrittenValues(EquationOp equation)
{
  auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());
  assert(terminator.lhs().size() == 1);
  assert(terminator.rhs().size() == 1);

  mlir::Value var = terminator.lhs()[0];

  while (var.isa<mlir::BlockArgument>())
  {
    mlir::Operation* op = var.getDefiningOp();
    assert(mlir::isa<LoadOp>(op) || mlir::isa<SubscriptionOp>(op));

    if (auto loadOp = mlir::dyn_cast<LoadOp>(op))
      var = loadOp.memory();
    else if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op))
      var = subscriptionOp.source();
  }

  assert(mlir::isa<ModelOp>(var.getParentRegion()->getParentOp()));
  return var;
}

static void getReadValues(mlir::Value value, llvm::SmallVectorImpl<mlir::Value>& variables)
{
  if (value.isa<mlir::BlockArgument>())
  {
    variables.push_back(value);
  }
  else
  {
    if (mlir::Operation* op = value.getDefiningOp())
    {
      for (mlir::Value operand : op->getOperands())
        getReadValues(operand, variables);
    }
  }
}

static void getReadVariables(EquationOp equation, llvm::SmallVectorImpl<mlir::Value>& variables)
{
  auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());
  assert(terminator.lhs().size() == 1);
  assert(terminator.rhs().size() == 1);
  getReadValues(terminator.rhs()[0], variables);
}

void VVarDependencyGraph::add(modelica::EquationOp equation)
{
  mlir::Value writtenVar = getWrittenValues(equation);

  llvm::SmallVector<mlir::Value, 3> reads;
  getReadVariables(equation, reads);

  
}
*/