#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "llvm/Support/ThreadPool.h"
#include <mutex>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  template<typename EquationOpKind>
  inline Equations<Equation> discoverEqs(mlir::modelica::ModelOp modelOp, const Variables& variables)
  {
    // Collect the operations to be mapped.
    llvm::SmallVector<EquationOpKind> ops;

    modelOp.getBodyRegion().walk([&](EquationOpKind op) {
      ops.push_back(op);
    });

    size_t numOfEquations = ops.size();

    // Map the equations.
    Equations<Equation> equations;
    equations.resize(numOfEquations);

    llvm::ThreadPool threadPool;

    // Function to map a chunk of equations.
    // 'from' index is included, 'to' index is excluded.
    auto mapFn = [&](size_t from, size_t to) {
      for (size_t i = from; i < to; ++i) {
        // No need for mutex locking, because the required space has already
        // been allocated and the accessed indices are independent by design.
        equations[i] = Equation::build(ops[i], variables);
      }
    };

    // Shard the work among multiple threads.
    unsigned int numOfThreads = threadPool.getThreadCount();
    size_t chunkSize = (numOfEquations + numOfThreads - 1) / numOfThreads;

    for (unsigned int i = 0; i < numOfThreads; ++i) {
      size_t from = i * chunkSize;
      size_t to = std::min(numOfEquations, (i + 1) * chunkSize);
      threadPool.async(mapFn, from, to);
    }

    // Wait for all the tasks to finish.
    threadPool.wait();

    assert(llvm::none_of(equations, [](const std::unique_ptr<Equation>& equation) {
             return equation == nullptr;
           }) && "Not all equations have been mapped");

    return equations;
  }
}

namespace marco::codegen
{
  Variables discoverVariables(ModelOp modelOp)
  {
    Variables variables;
    std::mutex mutex;

    mlir::ValueRange args = modelOp.getBodyRegion().getArguments();
    size_t numOfVariables = args.size();

    // Parallelize the variables mapping.
    llvm::ThreadPool threadPool;

    // Function to map a chunk of variables.
    // 'from' index is included, 'to' index is excluded.
    auto mapFn = [&](size_t from, size_t to) {
      for (size_t i = from; i < to; ++i) {
        std::lock_guard lockGuard(mutex);
        variables.add(Variable::build(args[i]));
      }
    };

    // Shard the work among multiple threads.
    unsigned int numOfThreads = threadPool.getThreadCount();
    size_t chunkSize = (numOfVariables + numOfThreads - 1) / numOfThreads;

    for (unsigned int i = 0; i < numOfThreads; ++i) {
      size_t from = i * chunkSize;
      size_t to = std::min(numOfVariables, (i + 1) * chunkSize);
      threadPool.async(mapFn, from, to);
    }

    // Wait for all the tasks to finish.
    threadPool.wait();

    assert(llvm::none_of(variables, [](const std::unique_ptr<Variable>& variable) {
             return variable == nullptr;
           }) && "Not all variables have been mapped");

    return variables;
  }

  Equations<Equation> discoverInitialEquations(mlir::modelica::ModelOp modelOp, const Variables& variables)
  {
    return discoverEqs<InitialEquationOp>(modelOp, variables);
  }

  Equations<Equation> discoverEquations(mlir::modelica::ModelOp modelOp, const Variables& variables)
  {
    return discoverEqs<EquationOp>(modelOp, variables);
  }

  namespace impl
  {
    BaseModel::BaseModel(mlir::modelica::ModelOp modelOp)
        : modelOp(modelOp.getOperation())
    {
    }

    BaseModel::~BaseModel() = default;

    ModelOp BaseModel::getOperation() const
    {
      return mlir::cast<ModelOp>(modelOp);
    }

    Variables BaseModel::getVariables() const
    {
      return variables;
    }

    void BaseModel::setVariables(Variables value)
    {
      this->variables = std::move(value);
      onVariablesSet(this->variables);
    }

    DerivativesMap& BaseModel::getDerivativesMap()
    {
      return derivativesMap;
    }

    const DerivativesMap& BaseModel::getDerivativesMap() const
    {
      return derivativesMap;
    }

    void BaseModel::setDerivativesMap(DerivativesMap map)
    {
      derivativesMap = std::move(map);
    }

    void BaseModel::onVariablesSet(Variables newVariables)
    {
      // Default implementation.
      // Do nothing.
    }
  }
}
