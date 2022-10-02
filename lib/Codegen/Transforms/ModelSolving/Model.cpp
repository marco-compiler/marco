#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "llvm/Support/ThreadPool.h"
#include <mutex>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  Variables discoverVariables(ModelOp modelOp)
  {
    Variables variables;

    mlir::ValueRange args = modelOp.getBodyRegion().getArguments();
    size_t numOfVariables = args.size();
    variables.resize(numOfVariables);

    // Parallelize the variables mapping.
    llvm::ThreadPool threadPool;

    // Function to map a chunk of variables.
    // 'from' index is included, 'to' index is excluded.
    auto mapFn = [&](size_t from, size_t to) {
      for (size_t i = from; i < to; ++i) {
        // No need for mutex locking, because the required space has already
        // been allocated and the accessed indices are independent by design.
        variables[i] = Variable::build(args[i]);
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
    Equations<Equation> result;

    modelOp.getBodyRegion().walk([&](InitialEquationOp equationOp) {
      result.add(Equation::build(equationOp, variables));
    });

    return result;
  }

  Equations<Equation> discoverEquations(mlir::modelica::ModelOp modelOp, const Variables& variables)
  {
    Equations<Equation> result;

    modelOp.getBodyRegion().walk([&](EquationOp equationOp) {
      result.add(Equation::build(equationOp, variables));
    });

    return result;
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
