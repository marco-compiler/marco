#define DEBUG_TYPE "runge-kutta"

#include "marco/Dialect/BaseModelica/Transforms/RungeKutta.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/Bridge.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "marco/Modeling/DependencyGraph.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_RUNGEKUTTAPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;
using namespace ::mlir::bmodelica::bridge;

namespace {
/// The class represents the Butcher tableau used in the Runge-Kutta method.
/// A tableau of order N has the form
///   c | A
///     | b'
///     | b*'
/// in which:
///   - A ia matrix of NxN elements (named "coefficients")
///   - b is a vector of N elements (named "weights")
///   - b* is a vector of N elements (named "adaptive weights")
///   - c is a vector of N elements (named "nodes")
class ButcherTableau {
public:
  /// @name Explicit Runge-Kutta
  /// {

  static std::unique_ptr<ButcherTableau> eulerForward();
  static std::unique_ptr<ButcherTableau> rk4();

private:
  static std::unique_ptr<ButcherTableau> genericMidPoint(double alpha);

public:
  static std::unique_ptr<ButcherTableau> midpoint();
  static std::unique_ptr<ButcherTableau> heun();
  static std::unique_ptr<ButcherTableau> ralston();

  /// }
  /// @name Explicit & adaptive Runge-Kutta
  /// {

  static std::unique_ptr<ButcherTableau> heunEuler();
  static std::unique_ptr<ButcherTableau> bogackiShampine();
  static std::unique_ptr<ButcherTableau> fehlberg();
  static std::unique_ptr<ButcherTableau> cashKarp();
  static std::unique_ptr<ButcherTableau> dormandPrince();

  /// }
  /// @name Implicit Runge-Kutta
  /// {

  static std::unique_ptr<ButcherTableau> eulerBackward();

  /// }

  ButcherTableau(int rows, int columns);

  /// Get the number of rows of the coefficients matrix.
  int getRows() const;

  /// Get the number of columns of the coefficients matrix.
  int getColumns() const;

  double getCoefficient(int row, int column) const;

  bool hasNonZeroOrderCoefficients(int row) const;

  void setCoefficients(llvm::ArrayRef<double> coefficients);

  double getWeight(int column) const;

  void setWeights(llvm::ArrayRef<double> weights);

  bool isAdaptive() const;

  double getAdaptiveWeight(int column) const;

  void setAdaptiveWeights(llvm::ArrayRef<double> adaptiveWeights);

  double getNode(int row) const;

  void setNodes(llvm::ArrayRef<double> nodes);

private:
  int rows;
  int columns;
  llvm::SmallVector<double, 16> coefficients;
  llvm::SmallVector<double, 4> weights;
  llvm::SmallVector<double, 4> adaptiveWeights;
  llvm::SmallVector<double, 4> nodes;
};
} // namespace

std::unique_ptr<ButcherTableau> ButcherTableau::eulerForward() {
  // 0  |
  // ---|---
  //    | 1

  auto tableau = std::make_unique<ButcherTableau>(1, 1);

  tableau->setCoefficients(0);
  tableau->setNodes(0);
  tableau->setWeights(1);

  return tableau;
}

std::unique_ptr<ButcherTableau> ButcherTableau::rk4() {
  // clang-format off

  // 0    |
  // 1/2  | 1/2
  // 1/2  | 0    1/2
  // 1    | 0    0     1
  // -----|---------------------
  //      | 1/6  1/3   1/3   1/6

  auto tableau = std::make_unique<ButcherTableau>(4, 4);

  tableau->setCoefficients({
    0,   0,   0, 0,
    0.5, 0,   0, 0,
    0,   0.5, 0, 0,
    0,   0,   1, 0
  });

  tableau->setNodes({0, 0.5, 0.5, 1});
  tableau->setWeights({1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6});

  return tableau;
  // clang-format on
}

std::unique_ptr<ButcherTableau> ButcherTableau::genericMidPoint(double alpha) {
  // clang-format off

  // 0      |
  // alpha  | alpha
  // -------|-----------------------------------------
  //        | 1 - 1 / (2 * alpha)     1 / (2 * alpha)

  auto tableau = std::make_unique<ButcherTableau>(2, 2);

  tableau->setCoefficients({
    0,     0,
    alpha, 0
  });

  tableau->setNodes({0, alpha});
  tableau->setWeights({1 - 1 / (2 * alpha), 1 / (2 * alpha)});

  return tableau;
  // clang-format on
}

std::unique_ptr<ButcherTableau> ButcherTableau::midpoint() {
  return genericMidPoint(0.5);
}

std::unique_ptr<ButcherTableau> ButcherTableau::heun() {
  return genericMidPoint(1);
}

std::unique_ptr<ButcherTableau> ButcherTableau::ralston() {
  return genericMidPoint(2.0 / 3);
}

std::unique_ptr<ButcherTableau> ButcherTableau::heunEuler() {
  // clang-format off

  // 0  | 0   0
  // 1  | 1   0
  // ---|-------
  //    | 1/2 1/2
  //    | 1   0

  auto tableau = std::make_unique<ButcherTableau>(2, 2);

  tableau->setCoefficients({
    0, 0,
    1, 0
  });

  tableau->setNodes({0, 1});
  tableau->setWeights({1.0 / 2, 1.0 / 2});
  tableau->setAdaptiveWeights({1, 0});

  return tableau;
  // clang-format on
}

std::unique_ptr<ButcherTableau> ButcherTableau::bogackiShampine() {
  // clang-format off

  // 0    |
  // 1/2  | 1/2
  // 3/4  | 0     3/4
  // 1    | 2/9   1/3   4/9
  // -----|-----------------------
  //      | 2/9   1/3   4/9   0
  //      | 7/24  1/4   1/3   1/8

  auto tableau = std::make_unique<ButcherTableau>(4, 4);

  tableau->setCoefficients({
    0,       0,       0,       0,
    1.0 / 2, 0,       0,       0,
    0,       3.0 / 4, 0,       0,
    2.0 / 9, 1.0 / 3, 4.0 / 9, 0
  });

  tableau->setNodes({0, 1.0 / 2, 3.0 / 4, 1});
  tableau->setWeights({2.0 / 9, 1.0 / 3, 4.0 / 9, 0});
  tableau->setAdaptiveWeights({7.0 / 24, 1.0 / 4, 1.0 / 3, 1.0 / 8});

  return tableau;
  // clang-format on
}

std::unique_ptr<ButcherTableau> ButcherTableau::fehlberg() {
  // clang-format off

  // 0      |
  // 1/4    | 1/4
  // 3/8    | 3/32        9/32
  // 12/13  | 1932/2197   -7200/2197    7296/2197
  // 1      | 439/216     -8            3680/513    -845/4104
  // 1/2    | -8/27       2             -3544/2565  1859/4104     -11/40
  // -------|------------------------------------------------------------------
  //        | 16/135      0             6656/12825  28561/56430   -9/50   2/55
  //        | 25/216      0             1408/2565   2197/4104     -1/5    0

  auto tableau = std::make_unique<ButcherTableau>(6, 6);

  tableau->setCoefficients({
    0,             0,              0,              0,             0,          0,
    1.0 / 4,       0,              0,              0,             0,          0,
    3.0 / 32,      9.0 / 32,       0,              0,             0,          0,
    1932.0 / 2197, -7200.0 / 2197, 7296.0 / 2197,  0,             0,          0,
    439.0 / 216,   -8,             3680.0 / 513,   -845.0 / 4104, 0,          0,
    -8.0 / 27,     2,              -3544.0 / 2565, 1859.0 / 4104, -11.0 / 40, 0
  });

  tableau->setNodes({0, 1.0 / 4, 3.0 / 8, 12.0 / 13, 1, 1.0 / 2});
  tableau->setWeights({16.0 / 135, 0, 6656.0 / 12825, 28561.0 / 56430, -9.0 / 50, 2.0 / 55});
  tableau->setAdaptiveWeights({25.0 / 216, 0, 1408.0 / 2565, 2197.0 / 4104, -1.0 / 5, 0});

  return tableau;
  // clang-format on
}

std::unique_ptr<ButcherTableau> ButcherTableau::cashKarp() {
  // clang-format off

  // 0    |
  // 1/5  | 1/5
  // 3/10 | 3/40        9/40
  // 3/5  | 3/10        -9/10    6/5
  // 1    | -11/54      5/2     -70/27      35/27
  // 7/8  | 1631/55296  175/512 575/13824   44275/110592  253/4096
  // -----|------------------------------------------------------------------
  //      | 37/378      0       250/521     125/594       0         512/1771
  //      | 2825/27648  0       18575/48384 13535/55296   277/14336 1/4

  auto tableau = std::make_unique<ButcherTableau>(6, 6);

  tableau->setCoefficients({
    0,              0,           0,             0,                0,            0,
    1.0 / 5,        0,           0,             0,                0,            0,
    3.0 / 40,       9.0 / 40,    0,             0,                0,            0,
    3.0 / 10,       -9.0 / 10,   6.0 / 5,       0,                0,            0,
    -11.0 / 54,     5.0 / 2,     -70.0 / 27,    35.0 / 27,        0,            0,
    1631.0 / 55296, 175.0 / 512, 575.0 / 13824, 44275.0 / 110592, 253.0 / 4096, 0
  });

  tableau->setNodes({0, 1.0 / 5, 3.0 / 10, 3.0 / 5, 1, 7.0 / 8});
  tableau->setWeights({37.0 / 378, 0, 250.0 / 521, 125.0 / 594, 0, 512.0 / 1771});
  tableau->setAdaptiveWeights({2825.0 / 27648, 0, 18575.0 / 48384, 13535.0 / 55296, 277.0 / 14336, 1.0 / 4});

  return tableau;
  // clang-format on
}

std::unique_ptr<ButcherTableau> ButcherTableau::dormandPrince() {
  // clang-format off

  // 0    |
  // 1/5  | 1/5
  // 3/10 | 3/40        9/40
  // 4/5  | 44/45       -56/15      32/9
  // 8/9  | 19372/6561  -25360/2187 64448/6561  -212/729
  // 1    | 9017/3168   -355/33     46732/5247  49/176    -5103/18656
  // 1    | 35/384      0           500/1113    125/192   -2187/6784    11/84
  // -----|--------------------------------------------------------------------
  //      | 35/384      0   500/1113    125/192   -2187/6784    11/84     0
  //      | 5179/57600  0   7571/16695  393/640   -92097/339200 187/2100  1/40

  auto tableau = std::make_unique<ButcherTableau>(7, 7);

  tableau->setCoefficients({
    0,              0,               0,              0,              0,               0,         0,
    1.0 / 5,        0,               0,              0,              0,               0,         0,
    3.0 / 40,       9.0 / 40,        0,              0,              0,               0,         0,
    44.0 / 45,      -56.0 / 15,      32.0 / 9,       0,              0,               0,         0,
    19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729,   0,               0,         0,
    9017.0 / 3168,  -355.0 / 33,     46732.0 / 5247, 49.0 / 176,     -5103.0 / 18656, 0,         0,
    35.0 / 384,     0,               500.0 / 1113,   125.0 / 192,    -2187.0 / 6784,  11.0 / 84, 0
  });

  tableau->setNodes({0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1, 1});
  tableau->setWeights({35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0});
  tableau->setAdaptiveWeights({5179.0 / 57600, 0, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200, 187.0 / 2100, 1.0 / 40});

  return tableau;
  // clang-format on
}

std::unique_ptr<ButcherTableau> ButcherTableau::eulerBackward() {
  // clang-format off

  // 1  | 1
  // ---|---
  //    | 1

  auto tableau = std::make_unique<ButcherTableau>(1, 1);

  tableau->setCoefficients(1);
  tableau->setNodes(1);
  tableau->setWeights(1);

  return tableau;
  // clang-format on
}

ButcherTableau::ButcherTableau(int rows, int columns)
    : rows(rows), columns(columns) {
  assert(rows > 0);
  assert(columns > 0);
  coefficients.resize(rows * columns, 0);
  weights.resize(columns, 0);
  adaptiveWeights.resize(columns, 0);
  nodes.resize(rows, 0);
}

int ButcherTableau::getRows() const { return rows; }

int ButcherTableau::getColumns() const { return columns; }

double ButcherTableau::getCoefficient(int row, int column) const {
  assert(row >= 1 && row <= rows);
  assert(column >= 1 && column <= columns);
  return coefficients[(row - 1) * columns + (column - 1)];
}

bool ButcherTableau::hasNonZeroOrderCoefficients(int row) const {
  for (int i = 1; i <= columns; ++i) {
    if (getCoefficient(row, i) != 0) {
      return true;
    }
  }

  return false;
}

void ButcherTableau::setCoefficients(llvm::ArrayRef<double> newCoefficients) {
  assert(newCoefficients.size() == static_cast<size_t>(rows * columns));
  coefficients.clear();
  coefficients.append(newCoefficients.begin(), newCoefficients.end());
}

double ButcherTableau::getWeight(int column) const {
  assert(column >= 1 && column <= columns);
  return weights[column - 1];
}

void ButcherTableau::setWeights(llvm::ArrayRef<double> newWeights) {
  assert(newWeights.size() == static_cast<size_t>(columns));
  weights.clear();
  weights.append(newWeights.begin(), newWeights.end());
}

bool ButcherTableau::isAdaptive() const {
  return llvm::any_of(adaptiveWeights,
                      [](double weight) { return weight != 0; });
}

double ButcherTableau::getAdaptiveWeight(int column) const {
  assert(column >= 1 && column <= columns);
  return adaptiveWeights[column - 1];
}

void ButcherTableau::setAdaptiveWeights(
    llvm::ArrayRef<double> newAdaptiveWeights) {
  assert(newAdaptiveWeights.size() == static_cast<size_t>(columns));
  adaptiveWeights.clear();
  adaptiveWeights.append(newAdaptiveWeights.begin(), newAdaptiveWeights.end());
}

double ButcherTableau::getNode(int row) const {
  assert(row >= 1 && row <= rows);
  return nodes[row - 1];
}

void ButcherTableau::setNodes(llvm::ArrayRef<double> newNodes) {
  assert(newNodes.size() == static_cast<size_t>(rows));
  nodes.clear();
  nodes.append(newNodes.begin(), newNodes.end());
}

// Given a variable of the model, returns the variable holding the next value.
using FutureVariablesMap = llvm::DenseMap<VariableOp, VariableOp>;

// Given a state variable of the model, returns the list of coefficients ('k')
// to be used for computing the next value.
using SlopeVariablesMap =
    llvm::DenseMap<VariableOp, llvm::SmallVector<VariableOp, 4>>;

// Given a state variable of the model, returns the variable holding the error
// for adaptive Runge-Kutta.
using ErrorVariablesMap = llvm::DenseMap<VariableOp, VariableOp>;

namespace {
struct EquationRhsFunction {
  IndexSet equationIndices;
  std::unique_ptr<AccessFunction> equationAccessFunction;
  FunctionOp functionOp;
};
} // namespace

// Given a state variable of the model, returns the function used to compute
// the right-hand side of the associated explicit ODE.
// The function is associated with the indices corresponding to the original
// equation.
using EquationFunctionsMap =
    llvm::DenseMap<VariableOp, llvm::SmallVector<EquationRhsFunction, 1>>;

namespace {
class RungeKuttaPass
    : public mlir::bmodelica::impl::RungeKuttaPassBase<RungeKuttaPass> {
public:
  using RungeKuttaPassBase<RungeKuttaPass>::RungeKuttaPassBase;

  void runOnOperation() override;

private:
  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  mlir::LogicalResult processModelOp(mlir::ModuleOp moduleOp, ModelOp modelOp);

  mlir::LogicalResult
  solveMainModel(mlir::IRRewriter &rewriter,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 mlir::ModuleOp moduleOp, ModelOp modelOp,
                 llvm::ArrayRef<SCCOp> SCCs);

  mlir::LogicalResult declareSupportVariables(
      mlir::IRRewriter &rewriter,
      mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
      const ButcherTableau &tableau, FutureVariablesMap &futureVars,
      SlopeVariablesMap &slopeVars, ErrorVariablesMap &errorVars);

  mlir::LogicalResult
  createEquationFunctions(mlir::IRRewriter &rewriter,
                          mlir::SymbolTableCollection &symbolTableCollection,
                          mlir::ModuleOp moduleOp, ModelOp modelOp,
                          const llvm::DenseSet<SCCOp> &derMatchedSCCs,
                          const FutureVariablesMap &futureVars,
                          EquationFunctionsMap &equationFunctions);

  /// Create the function for computing the right-hand side of the equation.
  std::pair<FunctionOp, std::unique_ptr<AccessFunction>>
  createEquationFunction(mlir::IRRewriter &rewriter,
                         mlir::SymbolTableCollection &symbolTableCollection,
                         mlir::ModuleOp moduleOp, ModelOp modelOp,
                         const FutureVariablesMap &futureVariables,
                         VariableOp stateVariableOp,
                         EquationInstanceOp equationOp);

  mlir::LogicalResult createTryStepFunction(
      mlir::IRRewriter &rewriter,
      mlir::SymbolTableCollection &symbolTableCollection,
      mlir::ModuleOp moduleOp, ModelOp modelOp,
      llvm::ArrayRef<VariableOp> stateVariableOps,
      llvm::ArrayRef<SCCOp> derMatchedSCCs, GlobalVariableOp timeStepVariableOp,
      const FutureVariablesMap &futureVars, const SlopeVariablesMap &slopeVars,
      const EquationFunctionsMap &eqRhsFuncs, const ButcherTableau &tableau);

  mlir::LogicalResult createEstimateErrorFunction(
      mlir::IRRewriter &rewriter,
      mlir::SymbolTableCollection &symbolTableCollection,
      mlir::ModuleOp moduleOp, ModelOp modelOp,
      llvm::ArrayRef<VariableOp> stateVariableOps,
      GlobalVariableOp timeStepVariableOp,
      const EquationFunctionsMap &eqRhsFuncs,
      const SlopeVariablesMap &slopeVars, const ErrorVariablesMap &errorVars,
      const ButcherTableau &tableau);

  mlir::LogicalResult
  createAcceptStepFunction(mlir::IRRewriter &rewriter,
                           mlir::SymbolTableCollection &symbolTableCollection,
                           mlir::ModuleOp moduleOp, ModelOp modelOp,
                           const FutureVariablesMap &futureVars);

  mlir::LogicalResult createUpdateNonStateVariablesFunction(
      mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp, ModelOp modelOp,
      llvm::ArrayRef<SCCOp> SCCs, const llvm::DenseSet<SCCOp> &derMatchedSCCs);

  mlir::LogicalResult
  computeSCCs(mlir::RewriterBase &rewriter,
              mlir::SymbolTableCollection &symbolTableCollection,
              ModelOp modelOp, DynamicOp dynamicOp,
              llvm::ArrayRef<EquationInstanceOp> equationOps);

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);
};
} // namespace

void RungeKuttaPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(moduleOp, [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  for (ModelOp modelOp : modelOps) {
    if (mlir::failed(processModelOp(moduleOp, modelOp))) {
      return signalPassFailure();
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return signalPassFailure();
    }
  }
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
RungeKuttaPass::getVariableAccessAnalysis(
    EquationTemplateOp equationTemplate,
    mlir::SymbolTableCollection &symbolTableCollection) {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::Operation *parentOp = equationTemplate->getParentOp();
  llvm::SmallVector<mlir::Operation *> parentOps;

  while (parentOp != moduleOp) {
    parentOps.push_back(parentOp);
    parentOp = parentOp->getParentOp();
  }

  mlir::AnalysisManager analysisManager = getAnalysisManager();

  for (mlir::Operation *op : llvm::reverse(parentOps)) {
    analysisManager = analysisManager.nest(op);
  }

  if (auto analysis =
          analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(
              equationTemplate)) {
    return *analysis;
  }

  auto &analysis = analysisManager.getChildAnalysis<VariableAccessAnalysis>(
      equationTemplate);

  if (mlir::failed(analysis.initialize(symbolTableCollection))) {
    return std::nullopt;
  }

  return std::reference_wrapper(analysis);
}

mlir::LogicalResult RungeKuttaPass::processModelOp(mlir::ModuleOp moduleOp,
                                                   ModelOp modelOp) {
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  llvm::SmallVector<SCCOp> mainSCCs;
  modelOp.collectMainSCCs(mainSCCs);

  llvm::SmallVector<VariableOp> variables;
  modelOp.collectVariables(variables);

  // Solve the 'main' model.
  if (mlir::failed(solveMainModel(rewriter, symbolTableCollection, moduleOp,
                                  modelOp, mainSCCs))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult RungeKuttaPass::solveMainModel(
    mlir::IRRewriter &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, llvm::ArrayRef<SCCOp> SCCs) {
  std::unique_ptr<ButcherTableau> tableau;

  if (variant == "euler-forward") {
    tableau = ButcherTableau::eulerForward();
  } else if (variant == "rk4") {
    tableau = ButcherTableau::rk4();
  } else if (variant == "midpoint") {
    tableau = ButcherTableau::midpoint();
  } else if (variant == "heun") {
    tableau = ButcherTableau::heun();
  } else if (variant == "ralston") {
    tableau = ButcherTableau::ralston();
  } else if (variant == "heun-euler") {
    tableau = ButcherTableau::heunEuler();
  } else if (variant == "bogacki-shampine") {
    tableau = ButcherTableau::bogackiShampine();
  } else if (variant == "fehlberg") {
    tableau = ButcherTableau::fehlberg();
  } else if (variant == "cash-karp") {
    tableau = ButcherTableau::cashKarp();
  } else if (variant == "dormand-prince") {
    tableau = ButcherTableau::dormandPrince();
  } else if (variant == "euler-backward") {
    tableau = ButcherTableau::eulerBackward();
  } else {
    tableau = ButcherTableau::rk4();
  }

  FutureVariablesMap futureVars;
  SlopeVariablesMap slopeVars;
  ErrorVariablesMap errorVars;
  EquationFunctionsMap equationFunctions;

  auto cleanOnFailure = llvm::make_scope_exit([&]() {
    for (const auto &entry : futureVars) {
      rewriter.eraseOp(entry.getSecond());
    }

    for (const auto &entry : slopeVars) {
      for (VariableOp variableOp : entry.getSecond()) {
        rewriter.eraseOp(variableOp);
      }
    }

    for (const auto &entry : errorVars) {
      rewriter.eraseOp(entry.getSecond());
    }

    for (const auto &entry : equationFunctions) {
      for (const auto &eqRhsFunc : entry.getSecond()) {
        rewriter.eraseOp(eqRhsFunc.functionOp);
      }
    }
  });

  // Create a global variable storing the value of the requested time step.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto timeStepVariableOp = rewriter.create<GlobalVariableOp>(
      modelOp.getLoc(), "timeStep",
      ArrayType::get(std::nullopt, rewriter.getF64Type()));

  symbolTableCollection.getSymbolTable(moduleOp).insert(timeStepVariableOp);

  // Collect the state variables.
  llvm::SmallVector<VariableOp> allVariableOps;
  llvm::SmallVector<VariableOp> stateVariableOps;
  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;
  modelOp.collectVariables(allVariableOps);

  for (VariableOp variableOp : allVariableOps) {
    if (derivativesMap.getDerivative(
            mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()))) {
      stateVariableOps.push_back(variableOp);
    }
  }

  // Collect the SCCs without cycles that have been matched with a derivative
  // variable. Fail if there is a cycle involving a derivative variable.
  llvm::SmallVector<SCCOp> derMatchedSCCs;
  llvm::DenseSet<SCCOp> derMatchedSCCsSet;

  for (SCCOp scc : SCCs) {
    // Collect the equations composing the SCC.
    llvm::SmallVector<EquationInstanceOp> equationOps;
    scc.collectEquations(equationOps);

    // Collect the matched variables.
    llvm::SmallVector<VariableOp> matchedVariables;

    for (EquationInstanceOp equationOp : equationOps) {
      matchedVariables.push_back(
          symbolTableCollection.lookupSymbolIn<VariableOp>(
              modelOp, equationOp.getProperties().match.name));
    }

    // Check if the SCC involves a modification of any state variable.
    bool involvesStateVariable =
        llvm::any_of(matchedVariables, [&](VariableOp variableOp) {
          auto variableName =
              mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());

          return derivativesMap.getDerivedVariable(variableName).has_value();
        });

    // Skip the SCC if no state variables is written.
    if (!involvesStateVariable) {
      continue;
    }

    // Fail in case of cycles involving state variables.
    if (scc.getCycle() && involvesStateVariable) {
      return mlir::failure();
    }

    derMatchedSCCs.push_back(scc);
    derMatchedSCCsSet.insert(scc);
  }

  // Declare the additional variables needed for the Runge-Kutta method.
  if (mlir::failed(declareSupportVariables(rewriter, symbolTableCollection,
                                           modelOp, *tableau, futureVars,
                                           slopeVars, errorVars))) {
    return mlir::failure();
  }

  // Create the functions for the right-hand side of the equations.
  if (mlir::failed(createEquationFunctions(rewriter, symbolTableCollection,
                                           moduleOp, modelOp, derMatchedSCCsSet,
                                           futureVars, equationFunctions))) {
    return mlir::failure();
  }

  if (mlir::failed(createTryStepFunction(
          rewriter, symbolTableCollection, moduleOp, modelOp, stateVariableOps,
          derMatchedSCCs, timeStepVariableOp, futureVars, slopeVars,
          equationFunctions, *tableau))) {
    return mlir::failure();
  }

  if (mlir::failed(createEstimateErrorFunction(
          rewriter, symbolTableCollection, moduleOp, modelOp, stateVariableOps,
          timeStepVariableOp, equationFunctions, slopeVars, errorVars,
          *tableau))) {
    return mlir::failure();
  }

  if (mlir::failed(createAcceptStepFunction(rewriter, symbolTableCollection,
                                            moduleOp, modelOp, futureVars))) {
    return mlir::failure();
  }

  if (mlir::failed(createUpdateNonStateVariablesFunction(
          rewriter, moduleOp, modelOp, SCCs, derMatchedSCCsSet))) {
    return mlir::failure();
  }

  // Erase the SCCs that have been handed over.
  for (SCCOp scc : derMatchedSCCs) {
    rewriter.eraseOp(scc);
  }

  cleanOnFailure.release();
  return mlir::success();
}

namespace {
VariableOp declareFutureVariable(mlir::OpBuilder &builder,
                                 VariableOp variableOp) {
  std::string name =
      getReservedVariableName("rk_" + variableOp.getSymName().str());

  auto variableType =
      VariableType::get(variableOp.getVariableType().getShape(),
                        RealType::get(builder.getContext()),
                        VariabilityProperty::none, IOProperty::none);

  return builder.create<VariableOp>(variableOp.getLoc(), name, variableType);
}

VariableOp declareSlopeVariable(mlir::OpBuilder &builder, VariableOp variableOp,
                                int order) {
  std::string name = getReservedVariableName(
      "rk_k" + std::to_string(order) + "_" + variableOp.getSymName().str());

  auto variableType =
      VariableType::get(variableOp.getVariableType().getShape(),
                        RealType::get(builder.getContext()),
                        VariabilityProperty::none, IOProperty::none);

  return builder.create<VariableOp>(variableOp.getLoc(), name, variableType);
}

VariableOp declareErrorVariable(mlir::OpBuilder &builder,
                                VariableOp variableOp) {
  std::string name =
      getReservedVariableName("rk_e_" + variableOp.getSymName().str());

  auto variableType =
      VariableType::get(variableOp.getVariableType().getShape(),
                        RealType::get(builder.getContext()),
                        VariabilityProperty::none, IOProperty::none);

  return builder.create<VariableOp>(variableOp.getLoc(), name, variableType);
}
} // namespace

mlir::LogicalResult RungeKuttaPass::declareSupportVariables(
    mlir::IRRewriter &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    const ButcherTableau &tableau, FutureVariablesMap &futureVars,
    SlopeVariablesMap &slopeVars, ErrorVariablesMap &errorVars) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(modelOp.getBody());

  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;

  for (VariableOp variableOp : modelOp.getVariables()) {
    if (auto derVar = derivativesMap.getDerivative(mlir::SymbolRefAttr::get(
            rewriter.getContext(), variableOp.getSymNameAttr()))) {
      // Declare the variable holding the next state value.
      VariableOp nextStateVar = declareFutureVariable(rewriter, variableOp);
      symbolTableCollection.getSymbolTable(modelOp).insert(nextStateVar);
      futureVars[variableOp] = nextStateVar;

      // Declare the variable holding the next derivative value.
      VariableOp derVariableOp =
          symbolTableCollection.lookupSymbolIn<VariableOp>(modelOp, *derVar);

      VariableOp nextDerVar = declareFutureVariable(rewriter, derVariableOp);
      symbolTableCollection.getSymbolTable(modelOp).insert(nextDerVar);
      futureVars[derVariableOp] = nextDerVar;

      // Declare the slope variables.
      for (int i = 1, rows = tableau.getRows(); i <= rows; ++i) {
        VariableOp slopeVar = declareSlopeVariable(rewriter, variableOp, i);
        symbolTableCollection.getSymbolTable(modelOp).insert(slopeVar);
        slopeVars[variableOp].push_back(slopeVar);
      }

      // Declare the error variables.
      if (tableau.isAdaptive()) {
        VariableOp errorVar = declareErrorVariable(rewriter, variableOp);
        symbolTableCollection.getSymbolTable(modelOp).insert(errorVar);
        errorVars[variableOp] = errorVar;
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult RungeKuttaPass::createEquationFunctions(
    mlir::IRRewriter &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, const llvm::DenseSet<SCCOp> &derMatchedSCCs,
    const FutureVariablesMap &futureVariables,
    EquationFunctionsMap &eqRhsFuncs) {
  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;

  for (SCCOp scc : derMatchedSCCs) {
    llvm::SmallVector<EquationInstanceOp> equationOps;
    scc.collectEquations(equationOps);
    assert(equationOps.size() == 1);

    for (EquationInstanceOp equationOp : equationOps) {
      auto stateVariable = derivativesMap.getDerivedVariable(
          equationOp.getProperties().match.name);

      if (!stateVariable) {
        return mlir::failure();
      }

      VariableOp stateVariableOp =
          symbolTableCollection.lookupSymbolIn<VariableOp>(modelOp,
                                                           *stateVariable);

      auto equationFunction = createEquationFunction(
          rewriter, symbolTableCollection, moduleOp, modelOp, futureVariables,
          stateVariableOp, equationOp);

      if (!equationFunction.first || !equationFunction.second) {
        return mlir::failure();
      }

      eqRhsFuncs[stateVariableOp].push_back({equationOp.getIterationSpace(),
                                             std::move(equationFunction.second),
                                             equationFunction.first});
    }
  }

  return mlir::success();
}

std::pair<FunctionOp, std::unique_ptr<AccessFunction>>
RungeKuttaPass::createEquationFunction(
    mlir::IRRewriter &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, const FutureVariablesMap &futureVariables,
    VariableOp stateVariableOp, EquationInstanceOp equationOp) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  EquationInstanceOp explicitEquationOp =
      equationOp.cloneAndExplicitate(rewriter, symbolTableCollection);

  if (!explicitEquationOp) {
    return {nullptr, nullptr};
  }

  auto eraseExplicitEquation = llvm::make_scope_exit([&]() {
    EquationTemplateOp templateOp = explicitEquationOp.getTemplate();
    rewriter.eraseOp(explicitEquationOp);
    rewriter.eraseOp(templateOp);
  });

  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp =
      rewriter.create<FunctionOp>(explicitEquationOp.getLoc(), "rk_eq");

  symbolTableCollection.getSymbolTable(moduleOp).insert(functionOp);
  rewriter.createBlock(&functionOp.getBodyRegion());

  // Declare the variables.
  rewriter.setInsertionPointToStart(functionOp.getBody());

  auto mappedTimeVariableOp = rewriter.create<VariableOp>(
      functionOp.getLoc(), "time",
      VariableType::get(std::nullopt, RealType::get(rewriter.getContext()),
                        VariabilityProperty::none, IOProperty::input));

  auto mappedStateVariableOp = rewriter.create<VariableOp>(
      functionOp.getLoc(), stateVariableOp.getSymName(),
      stateVariableOp.getVariableType().withIOProperty(IOProperty::input));

  llvm::SmallVector<VariableOp, 3> inductionVariablesOps;
  size_t numOfInductions = explicitEquationOp.getInductionVariables().size();

  for (size_t i = 0; i < numOfInductions; ++i) {
    std::string variableName = "ind" + std::to_string(i);

    auto variableType = VariableType::wrap(
        rewriter.getIndexType(), VariabilityProperty::none, IOProperty::input);

    auto variableOp = rewriter.create<VariableOp>(functionOp.getLoc(),
                                                  variableName, variableType);

    inductionVariablesOps.push_back(variableOp);
  }

  auto resultVariableOp = rewriter.create<VariableOp>(
      functionOp.getLoc(), "result",
      VariableType::get(std::nullopt, RealType::get(rewriter.getContext()),
                        VariabilityProperty::none, IOProperty::output));

  // Create the algorithm.
  auto algorithmOp = rewriter.create<AlgorithmOp>(functionOp.getLoc());
  rewriter.createBlock(&algorithmOp.getBodyRegion());
  rewriter.setInsertionPointToStart(algorithmOp.getBody());

  mlir::IRMapping mapping;
  llvm::DenseSet<VariableGetOp> mappedInductions;

  // Get the values of the induction variables.
  auto originalInductions = explicitEquationOp.getInductionVariables();

  for (size_t i = 0, e = originalInductions.size(); i < e; ++i) {
    auto mappedInduction = rewriter.create<VariableGetOp>(
        inductionVariablesOps[i].getLoc(),
        inductionVariablesOps[i].getVariableType().unwrap(),
        inductionVariablesOps[i].getSymName());

    mappedInductions.insert(mappedInduction);
    mapping.map(originalInductions[i], mappedInduction);
  }

  // Determine the operations to be cloned by starting from the terminator and
  // walking through the dependencies.
  llvm::DenseSet<mlir::Operation *> toBeCloned;
  llvm::SmallVector<mlir::Operation *> toBeClonedVisitStack;

  auto equationSidesOp = mlir::cast<EquationSidesOp>(
      explicitEquationOp.getTemplate().getBody()->getTerminator());

  mlir::Value rhs = equationSidesOp.getRhsValues()[0];

  if (mlir::Operation *rhsOp = rhs.getDefiningOp()) {
    toBeClonedVisitStack.push_back(rhsOp);
  }

  while (!toBeClonedVisitStack.empty()) {
    mlir::Operation *op = toBeClonedVisitStack.pop_back_val();
    toBeCloned.insert(op);

    for (mlir::Value operand : op->getOperands()) {
      if (auto *operandOp = operand.getDefiningOp()) {
        toBeClonedVisitStack.push_back(operandOp);
      }
    }

    op->walk([&](mlir::Operation *nestedOp) { toBeCloned.insert(nestedOp); });
  }

  // Clone the original operations and compute the residual value.
  for (auto &op : explicitEquationOp.getTemplate().getOps()) {
    if (!toBeCloned.contains(&op)) {
      continue;
    }

    if (mlir::isa<EquationSideOp, EquationSidesOp>(op)) {
      continue;
    }

    rewriter.clone(op, mapping);
  }

  rewriter.create<VariableSetOp>(explicitEquationOp.getLoc(), resultVariableOp,
                                 mapping.lookup(rhs));

  // Replace the access to the time and variables.
  llvm::SmallVector<TimeOp> timeOps;
  llvm::SmallVector<VariableGetOp> variableGetOps;

  algorithmOp->walk([&](mlir::Operation *nestedOp) {
    if (auto timeOp = mlir::dyn_cast<TimeOp>(nestedOp)) {
      timeOps.push_back(timeOp);
    }

    if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(nestedOp)) {
      variableGetOps.push_back(variableGetOp);
    }
  });

  for (TimeOp timeOp : timeOps) {
    rewriter.setInsertionPoint(timeOp);
    rewriter.replaceOpWithNewOp<VariableGetOp>(timeOp, mappedTimeVariableOp);
  }

  for (VariableGetOp variableGetOp : variableGetOps) {
    if (mappedInductions.contains(variableGetOp)) {
      // Skip the variables that have been introduced to map the original
      // inductions.
      continue;
    }

    rewriter.setInsertionPoint(variableGetOp);

    if (variableGetOp.getVariable() == mappedStateVariableOp.getSymName()) {
      rewriter.replaceOpWithNewOp<VariableGetOp>(variableGetOp,
                                                 mappedStateVariableOp);
    } else {
      VariableOp variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, variableGetOp.getVariableAttr());

      assert(variableOp && "Variable not found");
      auto futureVariableIt = futureVariables.find(variableOp);

      if (futureVariableIt == futureVariables.end()) {
        rewriter.replaceOpWithNewOp<QualifiedVariableGetOp>(variableGetOp,
                                                            variableOp);
      } else {
        rewriter.replaceOpWithNewOp<QualifiedVariableGetOp>(
            variableGetOp, futureVariableIt->getSecond());
      }
    }
  }

  auto lhsAccess = explicitEquationOp.getAccessAtPath(
      symbolTableCollection, EquationPath(EquationPath::LEFT, 0));

  return {functionOp, lhsAccess->getAccessFunction().clone()};
}

namespace {
mlir::LogicalResult
createEquationInstances(llvm::SmallVectorImpl<EquationInstanceOp> &newEquations,
                        mlir::RewriterBase &rewriter, mlir::Location loc,
                        mlir::SymbolTableCollection &symbolTableCollection,
                        DynamicOp dynamicOp, EquationTemplateOp templateOp,
                        const IndexSet &indices, bool wrapWithSCC = false) {
  rewriter.setInsertionPointToEnd(dynamicOp.getBody());

  if (wrapWithSCC) {
    rewriter.setInsertionPointToEnd(dynamicOp.getBody());
    auto sccOp = rewriter.create<SCCOp>(loc);

    rewriter.setInsertionPointToStart(
        rewriter.createBlock(&sccOp.getBodyRegion()));
  }

  auto instanceOp = rewriter.create<EquationInstanceOp>(loc, templateOp);

  std::optional<VariableAccess> access = instanceOp.getAccessAtPath(
      symbolTableCollection, EquationPath(EquationPath::EquationSide::LEFT, 0));

  newEquations.push_back(instanceOp);

  if (!access) {
    return mlir::failure();
  }

  instanceOp.getProperties().match = Variable(indices, *access);
  instanceOp.getProperties().setIndices(indices);

  return mlir::success();
}

/// Create the sum_(j = 1 to stages)(a_ij * k_j), with i = current order.
mlir::Value createSlopeSum(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ValueRange variableIndices,
                           llvm::ArrayRef<VariableOp> slopeVars,
                           const ButcherTableau &tableau, int currentOrder) {
  mlir::Value sum =
      builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));

  for (int i = 1, columns = tableau.getColumns(); i <= columns; ++i) {
    double tableauCoefficientValue = tableau.getCoefficient(currentOrder, i);

    if (tableauCoefficientValue == 0) {
      continue;
    }

    mlir::Value tableauCoefficient = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), tableauCoefficientValue));

    mlir::Value otherStageCoefficient =
        builder.create<VariableGetOp>(loc, slopeVars[i - 1]);

    if (!variableIndices.empty()) {
      otherStageCoefficient = builder.create<TensorExtractOp>(
          loc, otherStageCoefficient, variableIndices);
    }

    mlir::Value sumElement =
        builder.create<MulOp>(loc, RealType::get(builder.getContext()),
                              tableauCoefficient, otherStageCoefficient);

    sum = builder.create<AddOp>(loc, RealType::get(builder.getContext()), sum,
                                sumElement);
  }

  return sum;
}

mlir::LogicalResult
createSlopeEquation(llvm::SmallVectorImpl<EquationInstanceOp> &newEquations,
                    mlir::RewriterBase &rewriter, mlir::Location loc,
                    mlir::SymbolTableCollection &symbolTableCollection,
                    ModelOp modelOp, DynamicOp dynamicOp,
                    GlobalVariableOp timeStepVariable, VariableOp stateVariable,
                    const EquationRhsFunction &eqRhsFunc,
                    llvm::ArrayRef<VariableOp> slopeVars,
                    const ButcherTableau &tableau, int currentOrder) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the equation template.
  rewriter.setInsertionPointToEnd(modelOp.getBody());

  auto templateOp = rewriter.create<EquationTemplateOp>(loc);
  templateOp.createBody(eqRhsFunc.equationIndices.rank());
  rewriter.setInsertionPointToStart(templateOp.getBody());

  // Map the equation indices to the variable indices.
  llvm::SmallVector<mlir::Value> variableIndices;

  assert(eqRhsFunc.equationAccessFunction->isAffine() &&
         "Non-affine access detected");

  if (mlir::failed(materializeAffineMap(
          rewriter, templateOp.getLoc(),
          eqRhsFunc.equationAccessFunction->getAffineMap(),
          templateOp.getInductionVariables(), variableIndices))) {
    return mlir::failure();
  }

  // Get the value of the current order slope.
  mlir::Value lhs =
      rewriter.create<VariableGetOp>(loc, slopeVars[currentOrder - 1]);

  if (!variableIndices.empty()) {
    lhs = rewriter.create<TensorExtractOp>(loc, lhs, variableIndices);
  }

  // Compute the arguments to the call to the right-hand side equation function.
  llvm::SmallVector<mlir::Value, 2> callArgs;

  mlir::Value timeStep =
      rewriter.create<GlobalVariableGetOp>(loc, timeStepVariable);

  timeStep = rewriter.create<LoadOp>(timeStep.getLoc(), timeStep, std::nullopt);

  // Compute the first argument of the function call.
  // t + c_i * h, with i = current order
  mlir::Value timeOp = rewriter.create<TimeOp>(loc);
  mlir::Value firstArg = timeOp;

  if (auto nodeValue = tableau.getNode(currentOrder); nodeValue != 0) {
    mlir::Value node = rewriter.create<ConstantOp>(
        loc, RealAttr::get(rewriter.getContext(), nodeValue));

    mlir::Value mulNodeTimeStep = rewriter.create<MulOp>(
        loc, RealType::get(rewriter.getContext()), node, timeStep);

    firstArg = rewriter.create<AddOp>(loc, RealType::get(rewriter.getContext()),
                                      timeOp, mulNodeTimeStep);
  }

  callArgs.push_back(firstArg);

  // Compute the second argument of the function call.
  // state + h * sum_(i = 1 to stages)(a_ij * k_i), with i = current order
  mlir::Value var = rewriter.create<VariableGetOp>(loc, stateVariable);
  mlir::Value secondArg = var;

  if (tableau.hasNonZeroOrderCoefficients(currentOrder)) {
    mlir::Value scalarVar = var;

    if (!variableIndices.empty()) {
      scalarVar = rewriter.create<TensorExtractOp>(loc, var, variableIndices);
    }

    mlir::Value sum = createSlopeSum(rewriter, loc, variableIndices, slopeVars,
                                     tableau, currentOrder);

    mlir::Value mulTimeStepSum = rewriter.create<MulOp>(
        loc, RealType::get(rewriter.getContext()), timeStep, sum);

    mlir::Value increasedScalarVar = rewriter.create<AddOp>(
        loc, RealType::get(rewriter.getContext()), scalarVar, mulTimeStepSum);

    if (!variableIndices.empty()) {
      secondArg = rewriter.create<TensorInsertOp>(loc, increasedScalarVar,
                                                  secondArg, variableIndices);
    } else {
      secondArg = increasedScalarVar;
    }
  }

  callArgs.push_back(secondArg);
  llvm::append_range(callArgs, templateOp.getInductionVariables());

  auto callOp = rewriter.create<CallOp>(loc, eqRhsFunc.functionOp, callArgs);
  mlir::Value rhs = callOp.getResult(0);

  // Create the equation sides.
  lhs = rewriter.create<EquationSideOp>(loc, lhs);
  rhs = rewriter.create<EquationSideOp>(loc, rhs);
  rewriter.create<EquationSidesOp>(loc, lhs, rhs);

  // Create the equation instances.
  if (mlir::failed(createEquationInstances(
          newEquations, rewriter, loc, symbolTableCollection, dynamicOp,
          templateOp, eqRhsFunc.equationIndices))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult createSlopeEquations(
    llvm::SmallVectorImpl<EquationInstanceOp> &newEquations,
    mlir::RewriterBase &rewriter, mlir::Location loc,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    DynamicOp dynamicOp, GlobalVariableOp timeStepVariable,
    VariableOp stateVariableOp, llvm::ArrayRef<EquationRhsFunction> eqRhsFuncs,
    llvm::ArrayRef<VariableOp> slopeVars, const ButcherTableau &tableau) {
  for (int i = 1, rows = tableau.getRows(); i <= rows; ++i) {
    for (const auto &eqRhsFunc : eqRhsFuncs) {
      if (mlir::failed(createSlopeEquation(
              newEquations, rewriter, loc, symbolTableCollection, modelOp,
              dynamicOp, timeStepVariable, stateVariableOp, eqRhsFunc,
              slopeVars, tableau, i))) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

/// Create the sum_(i = 1 to stages)(b_i * k_i)
mlir::Value createFutureValueSum(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::ValueRange variableIndices,
                                 llvm::ArrayRef<VariableOp> slopeVars,
                                 const ButcherTableau &tableau) {
  mlir::Value sum =
      builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));

  for (int i = 1, columns = tableau.getColumns(); i <= columns; ++i) {
    double tableauWeightValue = tableau.getWeight(i);

    if (tableauWeightValue == 0) {
      continue;
    }

    mlir::Value tableauWeight = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), tableauWeightValue));

    mlir::Value slope = builder.create<VariableGetOp>(loc, slopeVars[i - 1]);

    if (!variableIndices.empty()) {
      slope = builder.create<TensorExtractOp>(loc, slope, variableIndices);
    }

    mlir::Value sumElement = builder.create<MulOp>(
        loc, RealType::get(builder.getContext()), tableauWeight, slope);

    sum = builder.create<AddOp>(loc, RealType::get(builder.getContext()), sum,
                                sumElement);
  }

  return sum;
}

mlir::LogicalResult createFutureStateValueEquation(
    llvm::SmallVectorImpl<EquationInstanceOp> &newEquations,
    mlir::RewriterBase &rewriter, mlir::Location loc,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    DynamicOp dynamicOp, GlobalVariableOp timeStepVariable,
    VariableOp currentStateVariableOp, VariableOp futureStateVariableOp,
    const EquationRhsFunction &eqRhsFunc, llvm::ArrayRef<VariableOp> slopeVars,
    const ButcherTableau &tableau) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the equation template.
  rewriter.setInsertionPointToEnd(modelOp.getBody());

  auto templateOp = rewriter.create<EquationTemplateOp>(loc);
  templateOp.createBody(eqRhsFunc.equationIndices.rank());
  rewriter.setInsertionPointToStart(templateOp.getBody());

  // Map the equation indices to the variable indices.
  llvm::SmallVector<mlir::Value> variableIndices;

  assert(eqRhsFunc.equationAccessFunction->isAffine() &&
         "Non-affine access detected");

  if (mlir::failed(materializeAffineMap(
          rewriter, templateOp.getLoc(),
          eqRhsFunc.equationAccessFunction->getAffineMap(),
          templateOp.getInductionVariables(), variableIndices))) {
    return mlir::failure();
  }

  // Get the future value of the state variable.
  mlir::Value lhs = rewriter.create<VariableGetOp>(loc, futureStateVariableOp);

  if (!variableIndices.empty()) {
    lhs = rewriter.create<TensorExtractOp>(loc, lhs, variableIndices);
  }

  // Compute the future value of the state variable.
  mlir::Value currentState =
      rewriter.create<VariableGetOp>(loc, currentStateVariableOp);

  if (!variableIndices.empty()) {
    currentState =
        rewriter.create<TensorExtractOp>(loc, currentState, variableIndices);
  }

  mlir::Value timeStep =
      rewriter.create<GlobalVariableGetOp>(loc, timeStepVariable);

  timeStep = rewriter.create<LoadOp>(timeStep.getLoc(), timeStep, std::nullopt);

  mlir::Value sum =
      createFutureValueSum(rewriter, loc, variableIndices, slopeVars, tableau);

  mlir::Value mul = rewriter.create<MulOp>(
      loc, RealType::get(rewriter.getContext()), timeStep, sum);

  mlir::Value rhs = rewriter.create<AddOp>(
      loc, RealType::get(rewriter.getContext()), currentState, mul);

  // Create the equation sides.
  lhs = rewriter.create<EquationSideOp>(loc, lhs);
  rhs = rewriter.create<EquationSideOp>(loc, rhs);
  rewriter.create<EquationSidesOp>(loc, lhs, rhs);

  // Create the equation instances.
  if (mlir::failed(createEquationInstances(
          newEquations, rewriter, loc, symbolTableCollection, dynamicOp,
          templateOp, eqRhsFunc.equationIndices))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult createFutureStateValueEquations(
    llvm::SmallVectorImpl<EquationInstanceOp> &newEquations,
    mlir::RewriterBase &rewriter, mlir::Location loc,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    DynamicOp dynamicOp, GlobalVariableOp timeStepVariable,
    VariableOp currentStateVariableOp, VariableOp futureStateVariableOp,
    llvm::ArrayRef<EquationRhsFunction> eqRhsFuncs,
    llvm::ArrayRef<VariableOp> slopeVars, const ButcherTableau &tableau) {
  for (const auto &eqRhsFunc : eqRhsFuncs) {
    if (mlir::failed(createFutureStateValueEquation(
            newEquations, rewriter, loc, symbolTableCollection, modelOp,
            dynamicOp, timeStepVariable, currentStateVariableOp,
            futureStateVariableOp, eqRhsFunc, slopeVars, tableau))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

/// Create the sum_(i = 1 to stages)(b_i - (b*)_i * k_i)
mlir::Value createErrorSum(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ValueRange variableIndices,
                           llvm::ArrayRef<VariableOp> slopeVars,
                           const ButcherTableau &tableau) {
  mlir::Value sum =
      builder.create<ConstantOp>(loc, RealAttr::get(builder.getContext(), 0));

  for (int i = 1, columns = tableau.getColumns(); i <= columns; ++i) {
    double tableauWeightValue = tableau.getWeight(i);
    double tableauAdaptiveWeightValue = tableau.getAdaptiveWeight(i);
    double tableauWeightDiffValue =
        tableauWeightValue - tableauAdaptiveWeightValue;

    if (tableauWeightDiffValue == 0) {
      continue;
    }

    mlir::Value tableauWeightDiff = builder.create<ConstantOp>(
        loc, RealAttr::get(builder.getContext(), tableauWeightDiffValue));

    mlir::Value slope = builder.create<VariableGetOp>(loc, slopeVars[i - 1]);

    if (!variableIndices.empty()) {
      slope = builder.create<TensorExtractOp>(loc, slope, variableIndices);
    }

    mlir::Value sumElement = builder.create<MulOp>(
        loc, RealType::get(builder.getContext()), tableauWeightDiff, slope);

    sum = builder.create<AddOp>(loc, RealType::get(builder.getContext()), sum,
                                sumElement);
  }

  return sum;
}

mlir::LogicalResult createErrorEquation(
    llvm::SmallVectorImpl<EquationInstanceOp> &newEquations,
    mlir::RewriterBase &rewriter, mlir::Location loc,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    DynamicOp dynamicOp, GlobalVariableOp timeStepVariableOp,
    VariableOp errorVariableOp, const EquationRhsFunction &eqRhsFunc,
    llvm::ArrayRef<VariableOp> slopeVars, const ButcherTableau &tableau) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the equation template.
  rewriter.setInsertionPointToEnd(modelOp.getBody());

  auto templateOp = rewriter.create<EquationTemplateOp>(loc);
  templateOp.createBody(eqRhsFunc.equationIndices.rank());
  rewriter.setInsertionPointToStart(templateOp.getBody());

  // Map the equation indices to the variable indices.
  llvm::SmallVector<mlir::Value> variableIndices;

  assert(eqRhsFunc.equationAccessFunction->isAffine() &&
         "Non-affine access detected");

  if (mlir::failed(materializeAffineMap(
          rewriter, templateOp.getLoc(),
          eqRhsFunc.equationAccessFunction->getAffineMap(),
          templateOp.getInductionVariables(), variableIndices))) {
    return mlir::failure();
  }

  // Get the error variable.
  mlir::Value lhs = rewriter.create<VariableGetOp>(loc, errorVariableOp);

  if (!variableIndices.empty()) {
    lhs = rewriter.create<TensorExtractOp>(loc, lhs, variableIndices);
  }

  // Compute the error.
  mlir::Value timeStep =
      rewriter.create<GlobalVariableGetOp>(loc, timeStepVariableOp);

  timeStep = rewriter.create<LoadOp>(timeStep.getLoc(), timeStep, std::nullopt);

  mlir::Value sum =
      createErrorSum(rewriter, loc, variableIndices, slopeVars, tableau);

  mlir::Value rhs = rewriter.create<MulOp>(
      loc, RealType::get(rewriter.getContext()), timeStep, sum);

  // Create the equation sides.
  lhs = rewriter.create<EquationSideOp>(loc, lhs);
  rhs = rewriter.create<EquationSideOp>(loc, rhs);
  rewriter.create<EquationSidesOp>(loc, lhs, rhs);

  // Create the equation instances.
  if (mlir::failed(createEquationInstances(
          newEquations, rewriter, loc, symbolTableCollection, dynamicOp,
          templateOp, eqRhsFunc.equationIndices, true))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult createErrorEquations(
    llvm::SmallVectorImpl<EquationInstanceOp> &newEquations,
    mlir::RewriterBase &rewriter, mlir::Location loc,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    DynamicOp dynamicOp, GlobalVariableOp timeStepVariable,
    VariableOp errorVariableOp, llvm::ArrayRef<EquationRhsFunction> eqRhsFuncs,
    llvm::ArrayRef<VariableOp> slopeVars, const ButcherTableau &tableau) {
  for (const auto &eqRhsFunc : eqRhsFuncs) {
    if (mlir::failed(createErrorEquation(
            newEquations, rewriter, loc, symbolTableCollection, modelOp,
            dynamicOp, timeStepVariable, errorVariableOp, eqRhsFunc, slopeVars,
            tableau))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult
createAcceptEquation(llvm::SmallVectorImpl<EquationInstanceOp> &newEquations,
                     mlir::RewriterBase &rewriter, mlir::Location loc,
                     mlir::SymbolTableCollection &symbolTableCollection,
                     ModelOp modelOp, DynamicOp dynamicOp,
                     VariableOp currentVariableOp,
                     VariableOp futureVariableOp) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the equation template.
  rewriter.setInsertionPointToEnd(modelOp.getBody());

  auto templateOp = rewriter.create<EquationTemplateOp>(loc);
  templateOp.createBody(currentVariableOp.getVariableType().getRank());
  rewriter.setInsertionPointToStart(templateOp.getBody());

  // Get the current variable.
  mlir::Value lhs = rewriter.create<VariableGetOp>(loc, currentVariableOp);

  if (auto inductions = templateOp.getInductionVariables();
      !inductions.empty()) {
    lhs = rewriter.create<TensorExtractOp>(loc, lhs, inductions);
  }

  // Get the future variable.
  mlir::Value rhs = rewriter.create<VariableGetOp>(loc, futureVariableOp);

  if (auto inductions = templateOp.getInductionVariables();
      !inductions.empty()) {
    rhs = rewriter.create<TensorExtractOp>(loc, rhs, inductions);
  }

  // Create the equation sides.
  lhs = rewriter.create<EquationSideOp>(loc, lhs);
  rhs = rewriter.create<EquationSideOp>(loc, rhs);
  rewriter.create<EquationSidesOp>(loc, lhs, rhs);

  // Create the equation instances.
  IndexSet variableIndices = currentVariableOp.getIndices();

  if (mlir::failed(createEquationInstances(
          newEquations, rewriter, loc, symbolTableCollection, dynamicOp,
          templateOp, variableIndices, true))) {
    return mlir::failure();
  }

  return mlir::success();
}
} // namespace

mlir::LogicalResult RungeKuttaPass::createTryStepFunction(
    mlir::IRRewriter &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, llvm::ArrayRef<VariableOp> stateVariableOps,
    llvm::ArrayRef<SCCOp> derMatchedSCCs, GlobalVariableOp timeStepVariableOp,
    const FutureVariablesMap &futureVars, const SlopeVariablesMap &slopeVars,
    const EquationFunctionsMap &eqRhsFuncs, const ButcherTableau &tableau) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function running the schedule.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::runtime::FunctionOp>(
      modelOp.getLoc(), "tryStep",
      rewriter.getFunctionType(rewriter.getF64Type(), std::nullopt));

  mlir::Block *entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  // Set the time step in the global variable.
  mlir::Value timeStepArg = functionOp.getArgument(0);

  auto timeStepArray = rewriter.create<GlobalVariableGetOp>(
      timeStepArg.getLoc(), timeStepVariableOp);

  rewriter.create<StoreOp>(timeStepArg.getLoc(), timeStepArg, timeStepArray,
                           std::nullopt);

  // Create the schedule operation.
  rewriter.setInsertionPointToEnd(modelOp.getBody());

  auto scheduleOp = rewriter.create<ScheduleOp>(modelOp.getLoc(), "rk_step");
  rewriter.createBlock(&scheduleOp.getBodyRegion());
  rewriter.setInsertionPointToStart(scheduleOp.getBody());

  auto dynamicOp = rewriter.create<DynamicOp>(modelOp.getLoc());
  rewriter.createBlock(&dynamicOp.getBodyRegion());
  rewriter.setInsertionPointToStart(dynamicOp.getBody());

  // Create the equations.
  llvm::SmallVector<EquationInstanceOp> newEquations;

  auto eraseNewEquationsOnFailure = llvm::make_scope_exit([&]() {
    llvm::DenseSet<EquationTemplateOp> templateOps;

    for (EquationInstanceOp equationOp : newEquations) {
      templateOps.insert(equationOp.getTemplate());
      rewriter.eraseOp(equationOp);
    }

    for (EquationTemplateOp templateOp : templateOps) {
      rewriter.eraseOp(templateOp);
    }
  });

  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;

  for (VariableOp variableOp : stateVariableOps) {
    auto eqRhsFuncsIt = eqRhsFuncs.find(variableOp);
    auto slopeVarsIt = slopeVars.find(variableOp);
    auto futureVarsIt = futureVars.find(variableOp);

    if (eqRhsFuncsIt == eqRhsFuncs.end() || slopeVarsIt == slopeVars.end() ||
        futureVarsIt == futureVars.end()) {
      return mlir::failure();
    }

    if (mlir::failed(createSlopeEquations(
            newEquations, rewriter, functionOp.getLoc(), symbolTableCollection,
            modelOp, dynamicOp, timeStepVariableOp, variableOp,
            eqRhsFuncsIt->getSecond(), slopeVarsIt->getSecond(), tableau))) {
      return mlir::failure();
    }

    if (mlir::failed(createFutureStateValueEquations(
            newEquations, rewriter, functionOp.getLoc(), symbolTableCollection,
            modelOp, dynamicOp, timeStepVariableOp, variableOp,
            futureVarsIt->getSecond(), eqRhsFuncsIt->getSecond(),
            slopeVarsIt->getSecond(), tableau))) {
      return mlir::failure();
    }

    auto derivative = derivativesMap.getDerivative(
        mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()));

    if (!derivative) {
      return mlir::failure();
    }

    auto derivedIndices = derivativesMap.getDerivedIndices(
        mlir::SymbolRefAttr::get(variableOp.getSymNameAttr()));

    if (!derivedIndices) {
      return mlir::failure();
    }

    VariableOp derVariableOp =
        symbolTableCollection.lookupSymbolIn<VariableOp>(modelOp, *derivative);

    auto futureDerVarsIt = futureVars.find(derVariableOp);

    if (futureDerVarsIt == futureVars.end()) {
      return mlir::failure();
    }

    for (SCCOp scc : derMatchedSCCs) {
      llvm::SmallVector<EquationInstanceOp> equationOps;
      scc.collectEquations(equationOps);

      for (EquationInstanceOp equationOp : equationOps) {
        mlir::OpBuilder::InsertionGuard equationGuard(rewriter);
        rewriter.setInsertionPointToEnd(dynamicOp.getBody());

        auto clonedEquationOp =
            equationOp.cloneAndExplicitate(rewriter, symbolTableCollection);

        rewriter.moveOpBefore(clonedEquationOp.getOperation(),
                              dynamicOp.getBody(), dynamicOp.getBody()->end());

        newEquations.push_back(clonedEquationOp);

        llvm::SmallVector<TimeOp> timeOps;
        llvm::SmallVector<VariableGetOp> variableGetOps;

        clonedEquationOp.getTemplate()->walk([&](mlir::Operation *nestedOp) {
          if (auto timeOp = mlir::dyn_cast<TimeOp>(nestedOp)) {
            timeOps.push_back(timeOp);
          }

          if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(nestedOp)) {
            variableGetOps.push_back(variableGetOp);
          }
        });

        for (TimeOp timeOp : timeOps) {
          rewriter.setInsertionPoint(timeOp);

          mlir::Value timeStep = rewriter.create<GlobalVariableGetOp>(
              timeOp.getLoc(), timeStepVariableOp);

          timeStep = rewriter.create<LoadOp>(timeStep.getLoc(), timeStep,
                                             std::nullopt);

          mlir::Value increasedTime = rewriter.create<AddOp>(
              timeOp.getLoc(), RealType::get(rewriter.getContext()), timeOp,
              timeStep);

          rewriter.replaceAllUsesWith(timeOp, increasedTime);
        }

        for (VariableGetOp variableGetOp : variableGetOps) {
          rewriter.setInsertionPoint(variableGetOp);

          VariableOp accessedVariableOp =
              symbolTableCollection.lookupSymbolIn<VariableOp>(
                  modelOp, variableGetOp.getVariableAttr());

          auto futureVariableIt = futureVars.find(accessedVariableOp);

          if (futureVariableIt != futureVars.end()) {
            rewriter.replaceOpWithNewOp<VariableGetOp>(
                variableGetOp, futureVariableIt->getSecond());
          }
        }

        VariableOp matchedVariableOp =
            symbolTableCollection.lookupSymbolIn<VariableOp>(
                modelOp, clonedEquationOp.getProperties().match.name);

        if (auto futureVarIt = futureVars.find(matchedVariableOp);
            futureVarIt != futureVars.end()) {
          VariableOp futureVarOp = futureVarIt->getSecond();
          clonedEquationOp.getProperties().match.name =
              mlir::SymbolRefAttr::get(futureVarOp.getSymNameAttr());
        }
      }
    }
  }

  if (mlir::failed(computeSCCs(rewriter, symbolTableCollection, modelOp,
                               dynamicOp, newEquations))) {
    return mlir::failure();
  }

  // Call the schedule.
  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<RunScheduleOp>(modelOp.getLoc(), scheduleOp);
  rewriter.create<mlir::runtime::ReturnOp>(modelOp.getLoc(), std::nullopt);

  eraseNewEquationsOnFailure.release();
  return mlir::success();
}

mlir::LogicalResult RungeKuttaPass::createEstimateErrorFunction(
    mlir::IRRewriter &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, llvm::ArrayRef<VariableOp> stateVariableOps,
    GlobalVariableOp timeStepVariableOp, const EquationFunctionsMap &eqRhsFuncs,
    const SlopeVariablesMap &slopeVars, const ErrorVariablesMap &errorVars,
    const ButcherTableau &tableau) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function running the schedule.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::runtime::FunctionOp>(
      modelOp.getLoc(), "estimateError",
      rewriter.getFunctionType(std::nullopt, rewriter.getF64Type()));

  mlir::Block *entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  llvm::SmallVector<mlir::Value, 1> functionResults;

  if (!tableau.isAdaptive()) {
    // The error estimate can be computed only in case of adaptive Runge-Kutta
    // methods.
    functionResults.push_back(rewriter.create<ConstantOp>(
        functionOp.getLoc(), rewriter.getF64FloatAttr(0)));
  } else {
    // Create the schedule operation.
    rewriter.setInsertionPointToEnd(modelOp.getBody());

    auto scheduleOp = rewriter.create<ScheduleOp>(modelOp.getLoc(), "rk_error");
    rewriter.createBlock(&scheduleOp.getBodyRegion());
    rewriter.setInsertionPointToStart(scheduleOp.getBody());

    auto dynamicOp = rewriter.create<DynamicOp>(modelOp.getLoc());
    rewriter.createBlock(&dynamicOp.getBodyRegion());
    rewriter.setInsertionPointToStart(dynamicOp.getBody());

    // Create the equations.
    llvm::SmallVector<EquationInstanceOp> newEquations;

    auto eraseNewEquationsOnFailure = llvm::make_scope_exit([&]() {
      llvm::DenseSet<EquationTemplateOp> templateOps;

      for (EquationInstanceOp equationOp : newEquations) {
        templateOps.insert(equationOp.getTemplate());
        rewriter.eraseOp(equationOp);
      }

      for (EquationTemplateOp templateOp : templateOps) {
        rewriter.eraseOp(templateOp);
      }
    });

    for (VariableOp variableOp : stateVariableOps) {
      if (tableau.isAdaptive()) {
        auto errorVarsIt = errorVars.find(variableOp);
        auto eqRhsFuncIt = eqRhsFuncs.find(variableOp);
        auto slopeVarsIt = slopeVars.find(variableOp);

        if (errorVarsIt == errorVars.end() || eqRhsFuncIt == eqRhsFuncs.end() ||
            slopeVarsIt == slopeVars.end()) {
          return mlir::failure();
        }

        if (mlir::failed(createErrorEquations(
                newEquations, rewriter, functionOp.getLoc(),
                symbolTableCollection, modelOp, dynamicOp, timeStepVariableOp,
                errorVarsIt->getSecond(), eqRhsFuncIt->getSecond(),
                slopeVarsIt->getSecond(), tableau))) {
          return mlir::failure();
        }
      }
    }

    // Call the schedule.
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<RunScheduleOp>(modelOp.getLoc(), scheduleOp);

    // Compute the maximum error.
    mlir::Value error = rewriter.create<ConstantOp>(
        functionOp.getLoc(), RealAttr::get(rewriter.getContext(), 0));

    for (VariableOp variableOp : stateVariableOps) {
      auto errorVarsIt = errorVars.find(variableOp);

      if (errorVarsIt == errorVars.end()) {
        return mlir::failure();
      }

      mlir::Value variableError = rewriter.create<QualifiedVariableGetOp>(
          functionOp.getLoc(), errorVarsIt->getSecond());

      if (mlir::isa<mlir::TensorType>(variableError.getType())) {
        variableError =
            rewriter.create<MaxOp>(functionOp.getLoc(), variableError);
      }

      error = rewriter.create<MaxOp>(functionOp.getLoc(), error, variableError);
    }

    error = rewriter.create<CastOp>(functionOp.getLoc(), rewriter.getF64Type(),
                                    error);

    functionResults.push_back(error);
    eraseNewEquationsOnFailure.release();
  }

  rewriter.create<mlir::runtime::ReturnOp>(modelOp.getLoc(), functionResults);
  return mlir::success();
}

mlir::LogicalResult RungeKuttaPass::createAcceptStepFunction(
    mlir::IRRewriter &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, const FutureVariablesMap &futureVars) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function running the schedule.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::runtime::FunctionOp>(
      modelOp.getLoc(), "acceptStep",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block *entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  // Create the schedule operation.
  rewriter.setInsertionPointToEnd(modelOp.getBody());

  auto scheduleOp = rewriter.create<ScheduleOp>(modelOp.getLoc(), "rk_accept");
  rewriter.createBlock(&scheduleOp.getBodyRegion());
  rewriter.setInsertionPointToStart(scheduleOp.getBody());

  auto dynamicOp = rewriter.create<DynamicOp>(modelOp.getLoc());
  rewriter.createBlock(&dynamicOp.getBodyRegion());
  rewriter.setInsertionPointToStart(dynamicOp.getBody());

  // Create the equations.
  llvm::SmallVector<EquationInstanceOp> newEquations;

  auto eraseNewEquationsOnFailure = llvm::make_scope_exit([&]() {
    llvm::DenseSet<EquationTemplateOp> templateOps;

    for (EquationInstanceOp equationOp : newEquations) {
      templateOps.insert(equationOp.getTemplate());
      rewriter.eraseOp(equationOp);
    }

    for (EquationTemplateOp templateOp : templateOps) {
      rewriter.eraseOp(templateOp);
    }
  });

  llvm::SmallVector<VariableOp> variableOps;
  modelOp.collectVariables(variableOps);

  for (VariableOp variableOp : variableOps) {
    auto futureVarsIt = futureVars.find(variableOp);

    if (futureVarsIt == futureVars.end()) {
      continue;
    }

    if (mlir::failed(createAcceptEquation(
            newEquations, rewriter, functionOp.getLoc(), symbolTableCollection,
            modelOp, dynamicOp, variableOp, futureVarsIt->getSecond()))) {
      return mlir::failure();
    }
  }

  // Call the schedule.
  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<RunScheduleOp>(modelOp.getLoc(), scheduleOp);
  rewriter.create<mlir::runtime::ReturnOp>(modelOp.getLoc(), std::nullopt);

  eraseNewEquationsOnFailure.release();
  return mlir::success();
}

mlir::LogicalResult RungeKuttaPass::createUpdateNonStateVariablesFunction(
    mlir::RewriterBase &rewriter, mlir::ModuleOp moduleOp, ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs, const llvm::DenseSet<SCCOp> &derMatchedSCCs) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Create the function running the schedule.
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto functionOp = rewriter.create<mlir::runtime::FunctionOp>(
      modelOp.getLoc(), "updateNonStateVariables",
      rewriter.getFunctionType(std::nullopt, std::nullopt));

  mlir::Block *entryBlock = functionOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  llvm::SmallVector<SCCOp> nonDerMatchedSCCs;

  for (SCCOp scc : SCCs) {
    if (!derMatchedSCCs.contains(scc)) {
      nonDerMatchedSCCs.push_back(scc);
    }
  }

  if (!derMatchedSCCs.empty()) {
    rewriter.setInsertionPointToEnd(modelOp.getBody());

    // Create the schedule operation.
    auto scheduleOp = rewriter.create<ScheduleOp>(modelOp.getLoc(), "dynamic");
    rewriter.createBlock(&scheduleOp.getBodyRegion());
    rewriter.setInsertionPointToStart(scheduleOp.getBody());

    auto dynamicOp = rewriter.create<DynamicOp>(modelOp.getLoc());
    rewriter.createBlock(&dynamicOp.getBodyRegion());
    rewriter.setInsertionPointToStart(dynamicOp.getBody());

    for (SCCOp scc : SCCs) {
      scc->moveBefore(dynamicOp.getBody(), dynamicOp.getBody()->end());
    }

    // Call the schedule.
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<RunScheduleOp>(modelOp.getLoc(), scheduleOp);
  }

  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<mlir::runtime::ReturnOp>(modelOp.getLoc(), std::nullopt);
  return mlir::success();
}

mlir::LogicalResult
RungeKuttaPass::computeSCCs(mlir::RewriterBase &rewriter,
                            mlir::SymbolTableCollection &symbolTableCollection,
                            ModelOp modelOp, DynamicOp dynamicOp,
                            llvm::ArrayRef<EquationInstanceOp> equationOps) {
  auto storage = bridge::Storage::create();
  llvm::SmallVector<EquationBridge *> equationPtrs;

  for (VariableOp variableOp : modelOp.getVariables()) {
    storage->addVariable(variableOp);
  }

  for (EquationInstanceOp equation : equationOps) {
    auto &bridge = storage->addEquation(
        static_cast<int64_t>(storage->equationBridges.size()), equation,
        symbolTableCollection);

    if (auto accessAnalysis = getVariableAccessAnalysis(
            equation.getTemplate(), symbolTableCollection)) {
      bridge.setAccessAnalysis(*accessAnalysis);
    }

    equationPtrs.push_back(&bridge);
  }

  using DependencyGraph =
      marco::modeling::DependencyGraph<VariableBridge *, EquationBridge *>;

  DependencyGraph dependencyGraph(rewriter.getContext());
  dependencyGraph.addEquations(equationPtrs);

  llvm::SmallVector<DependencyGraph::SCC> SCCs = dependencyGraph.getSCCs();
  rewriter.setInsertionPointToEnd(dynamicOp.getBody());

  for (const DependencyGraph::SCC &scc : SCCs) {
    auto sccOp = rewriter.create<SCCOp>(modelOp.getLoc());
    mlir::OpBuilder::InsertionGuard sccGuard(rewriter);

    rewriter.setInsertionPointToStart(
        rewriter.createBlock(&sccOp.getBodyRegion()));

    for (const auto &sccElement : scc) {
      const auto &equation = dependencyGraph[*sccElement];
      const IndexSet &indices = sccElement.getIndices();

      size_t numOfInductions = equation->getOp().getInductionVariables().size();
      bool isScalarEquation = numOfInductions == 0;

      llvm::SmallVector<VariableAccess> accesses;
      llvm::SmallVector<VariableAccess> writeAccesses;

      if (mlir::failed(
              equation->getOp().getAccesses(accesses, symbolTableCollection))) {
        return mlir::failure();
      }

      if (mlir::failed(equation->getOp().getWriteAccesses(
              writeAccesses, symbolTableCollection, accesses))) {
        return mlir::failure();
      }

      llvm::sort(writeAccesses,
                 [](const VariableAccess &first, const VariableAccess &second) {
                   if (first.getAccessFunction().isAffine() &&
                       !second.getAccessFunction().isAffine()) {
                     return true;
                   }

                   if (!first.getAccessFunction().isAffine() &&
                       second.getAccessFunction().isAffine()) {
                     return false;
                   }

                   return first < second;
                 });

      auto clonedOp = mlir::cast<EquationInstanceOp>(
          rewriter.clone(*equation->getOp().getOperation()));

      IndexSet slicedIndices;

      if (!isScalarEquation) {
        slicedIndices = indices.takeFirstDimensions(numOfInductions);
      }

      if (mlir::failed(
              clonedOp.setIndices(slicedIndices, symbolTableCollection))) {
        return mlir::failure();
      }
    }
  }

  for (EquationInstanceOp equationOp : equationOps) {
    rewriter.eraseOp(equationOp);
  }

  return mlir::success();
}

mlir::LogicalResult RungeKuttaPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  mlir::GreedyRewriteConfig config;
  config.enableFolding();
  return mlir::applyPatternsGreedily(modelOp, std::move(patterns), config);
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createRungeKuttaPass() {
  return std::make_unique<RungeKuttaPass>();
}

std::unique_ptr<mlir::Pass>
createRungeKuttaPass(const RungeKuttaPassOptions &options) {
  return std::make_unique<RungeKuttaPass>(options);
}
} // namespace mlir::bmodelica
