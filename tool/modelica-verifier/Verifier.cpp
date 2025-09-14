#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/AllInterfaces.h"
#include "marco/JIT/EngineBuilder.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include <fstream>
#include <iostream>

#define TOLERANCE_DEFAULT 1e-9

using namespace ::mlir::bmodelica;

/// The map associates each equation to the information needed for the
/// computation of its left-hand and right-hand sides.
/// Each tuple associated to an equation is composed of:
///   - Function computing the left-hand side value.
///   - Function computing the right-hand side value.
///   - List of variables taken as arguments.
using EquationFunctionsMap =
    llvm::DenseMap<EquationInstanceOp,
                   std::tuple<mlir::func::FuncOp, mlir::func::FuncOp,
                              llvm::SmallVector<VariableOp>>>;

namespace marco::verifier {
void parseCSVHeader(
    llvm::StringRef line,
    llvm::SmallVectorImpl<std::pair<std::string, double>> &variables);

bool parseValues(
    llvm::StringRef line,
    llvm::SmallVectorImpl<std::pair<std::string, double>> &variables);

llvm::DenseMap<VariableOp, StridedMemRefType<double, 1>>
createMemRefDescriptors(ModelOp modelOp);

mlir::LogicalResult
createEquationFunctions(mlir::ModuleOp moduleOp, ModelOp modelOp,
                        llvm::ArrayRef<EquationInstanceOp> initialEquations,
                        llvm::ArrayRef<EquationInstanceOp> dynamicEquations,
                        EquationFunctionsMap &mapping);

bool hasHelpFlag(int argc, char *argv[]);

double getTolerance(int argc, char *argv[]);

void getLinkedLibs(int argc, char *argv[],
                   llvm::SmallVectorImpl<llvm::StringRef> &libs);

bool checkEquations(
    mlir::ExecutionEngine &executionEngine, ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    llvm::ArrayRef<EquationInstanceOp> equations,
    const EquationFunctionsMap &equationFunctions,
    llvm::DenseMap<VariableOp, StridedMemRefType<double, 1>> &variableMemRefs,
    llvm::ArrayRef<std::pair<std::string, double>> variables, double tolerance);
} // namespace marco::verifier

using namespace ::marco::verifier;
using namespace ::marco::jit;

void printHelp(char *argv0) {
  std::cout << "Usage: " << argv0
            << " <model.mlir> <data.csv> [--tolerance=<value>] [-l lib]..."
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (hasHelpFlag(argc - 1, argv + 1)) {
    printHelp(argv[0]);
    return EXIT_SUCCESS;
  }

  if (argc < 3) {
    printHelp(argv[0]);
    return EXIT_FAILURE;
  }

  mlir::DialectRegistry mlirDialectRegistry;

  // Register dialects.
  mlir::registerAllDialects(mlirDialectRegistry);
  mlirDialectRegistry.insert<BaseModelicaDialect>();

  // Register extensions.
  mlir::registerAllExtensions(mlirDialectRegistry);

  mlir::bmodelica::registerAllDialectInterfaceImplementations(
      mlirDialectRegistry);

  // Create the MLIR context.
  auto mlirContext = std::make_unique<mlir::MLIRContext>(mlirDialectRegistry);
  mlirContext->loadAllAvailableDialects();

  // Register translations to LLVM-IR.
  mlir::registerBuiltinDialectTranslation(*mlirContext);
  mlir::registerLLVMDialectTranslation(*mlirContext);

  // Parse the MLIR file.
  llvm::SourceMgr sourceMgr;

  mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
      mlir::parseSourceFile<mlir::ModuleOp>(argv[1], sourceMgr,
                                            mlirContext.get());

  if (!moduleOp || mlir::failed(moduleOp->verifyInvariants())) {
    llvm::errs() << "Could not parse MLIR\n";
    return EXIT_FAILURE;
  }

  // Collect the initial and dynamic equations.
  llvm::SmallVector<ModelOp> modelOps;

  llvm::SmallVector<VariableOp> variableOps;
  llvm::SmallVector<EquationInstanceOp> initialEquations;
  llvm::SmallVector<EquationInstanceOp> dynamicEquations;

  moduleOp->walk([&](ModelOp modelOp) { modelOps.push_back(modelOp); });

  if (modelOps.empty()) {
    llvm::errs() << "No model found\n";
    return EXIT_FAILURE;
  }

  if (modelOps.size() > 1) {
    llvm::errs() << "Expected 1 model, found " << modelOps.size() << "\n";
    return EXIT_FAILURE;
  }

  ModelOp modelOp = modelOps[0];

  modelOp.collectVariables(variableOps);
  modelOp.collectInitialEquations(initialEquations);
  modelOp.collectMainEquations(dynamicEquations);

  // Open the CSV file.
  std::ifstream csv(argv[2]);

  if (!csv.is_open()) {
    llvm::errs() << "Error opening CSV file: " << argv[2] << "\n";
    return EXIT_FAILURE;
  }

  // Read the CSV file.
  std::string line;

  // Parse the header.
  if (!std::getline(csv, line)) {
    llvm::errs() << "Empty CSV file\n";
    return EXIT_FAILURE;
  }

  llvm::SmallVector<std::pair<std::string, double>> variables;

  parseCSVHeader(line, variables);

  // Create the left-hand side and right-hand functions for each equation.
  EquationFunctionsMap equationFunctions;

  if (mlir::failed(createEquationFunctions(*moduleOp, modelOp, initialEquations,
                                           dynamicEquations,
                                           equationFunctions))) {
    return EXIT_FAILURE;
  }

  // Create the memref descriptors for the variables.
  auto memRefDescriptors = createMemRefDescriptors(modelOp);

  // Check each step.
  mlir::ExecutionEngineOptions engineOptions;
  // engineOptions.enableObjectDump = true;

  double tolerance = getTolerance(argc - 3, argv + 3);

  llvm::SmallVector<llvm::StringRef> linkedLibs;
  getLinkedLibs(argc - 3, argv + 3, linkedLibs);
  engineOptions.sharedLibPaths = linkedLibs;

  EngineBuilder engineBuilder(*moduleOp, engineOptions);
  auto engine = engineBuilder.getEngine();
  // engine.get()->dumpToObjectFile("/home/mscuttari/test_verifier/dump.o");

  bool firstRow = true;

  if (!engine) {
    return EXIT_FAILURE;
  }

  while (std::getline(csv, line)) {
    if (!parseValues(line, variables)) {
      return EXIT_FAILURE;
    }

    if (firstRow) {
      if (!checkEquations(*engine, modelOp, variableOps, initialEquations,
                          equationFunctions, memRefDescriptors, variables,
                          tolerance)) {
        return EXIT_FAILURE;
      }

      firstRow = false;
    } else {
      if (!checkEquations(*engine, modelOp, variableOps, dynamicEquations,
                          equationFunctions, memRefDescriptors, variables,
                          tolerance)) {
        return EXIT_FAILURE;
      }
    }
  }

  csv.close();

  // Deallocate the memory for the variables.
  for (auto &memRefDescriptor : memRefDescriptors) {
    delete[] memRefDescriptor.second.basePtr;
  }

  llvm::outs() << "Data is consistent\n";
  return EXIT_SUCCESS;
}

namespace marco::verifier {
void parseCSVHeader(
    llvm::StringRef line,
    llvm::SmallVectorImpl<std::pair<std::string, double>> &variables) {
  std::string value;
  bool insideQuotes = false;
  std::string field;

  for (char ch : line) {
    if (ch == '"') {
      insideQuotes = !insideQuotes;
    } else if (ch == ',' && !insideQuotes) {
      variables.push_back({field, 0xDEADBEEF});
      field.clear();
    } else {
      field += ch;
    }
  }

  if (!field.empty()) {
    variables.push_back({field, 0xDEADBEEF});
  }
}

bool parseValues(
    llvm::StringRef line,
    llvm::SmallVectorImpl<std::pair<std::string, double>> &variables) {
  llvm::SmallVector<llvm::StringRef, 8> fields;
  line.split(fields, ',');

  for (auto field : llvm::enumerate(fields)) {
    double value;

    if (!field.value().getAsDouble(value)) {
      variables[field.index()].second = value;
    } else {
      llvm::errs() << "Invalid double value: " << field.value() << "\n";
      return false;
    }
  }

  return true;
}

llvm::DenseMap<VariableOp, StridedMemRefType<double, 1>>
createMemRefDescriptors(ModelOp modelOp) {
  llvm::DenseMap<VariableOp, StridedMemRefType<double, 1>> result;

  modelOp.walk([&](VariableOp variableOp) {
    size_t flatSize = variableOp.getIndices().flatSize();
    double *data = new double[flatSize];

    for (size_t i = 0; i < flatSize; ++i) {
      data[i] = 0xDEADBEEF;
    }

    result[variableOp] =
        StridedMemRefType<double, 1>{.basePtr = data,
                                     .data = data,
                                     .offset = 0,
                                     .sizes = {static_cast<int64_t>(flatSize)},
                                     .strides = {1}};
  });

  return result;
}

mlir::func::FuncOp createEquationSideFunction(
    mlir::ModuleOp moduleOp, mlir::SymbolTableCollection &symbolTableCollection,
    llvm::StringRef name, EquationInstanceOp equationOp,
    llvm::ArrayRef<VariableOp> accessedVariables, EquationPath path) {
  mlir::OpBuilder builder(moduleOp.getContext());
  mlir::Location loc = equationOp.getLoc();

  // Collect the accessed variables.
  llvm::SmallVector<VariableGetOp> getOps;

  equationOp.walk([&](VariableGetOp getOp) { getOps.push_back(getOp); });

  // Determine the argument types.
  llvm::SmallVector<mlir::Type> inputTypes;

  // Time variable.
  inputTypes.push_back(builder.getF64Type());

  // Indices.
  size_t inductionsOffset = inputTypes.size();

  inputTypes.append(equationOp.getInductionVariables().size(),
                    builder.getIndexType());

  // Variables (unranked memref descriptors).
  size_t variablesOffset = inputTypes.size();

  inputTypes.append(accessedVariables.size(),
                    mlir::UnrankedMemRefType::get(builder.getF64Type(), 0));

  // Create the function.
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, name, builder.getFunctionType(inputTypes, builder.getF64Type()));

  symbolTableCollection.getSymbolTable(moduleOp).insert(funcOp);
  builder.setInsertionPointToStart(funcOp.addEntryBlock());

  mlir::IRMapping mapping;

  // Map the indices.
  for (auto inductionVariable :
       llvm::enumerate(equationOp.getInductionVariables())) {
    mapping.map(
        inductionVariable.value(),
        funcOp.getArgument(inductionsOffset + inductionVariable.index()));
  }

  // Extract the variable memrefs.
  llvm::StringMap<mlir::Value> variables;

  for (auto accessedVar : llvm::enumerate(accessedVariables)) {
    VariableOp variableOp = accessedVar.value();

    auto memRefType = mlir::MemRefType::get(
        variableOp.getVariableType().getShape(), builder.getF64Type());

    auto tensorType = mlir::RankedTensorType::get(
        variableOp.getVariableType().getShape(), builder.getF64Type());

    auto stridesAndOffset = memRefType.getStridesAndOffset();

    auto reinterpretOp = builder.create<mlir::memref::ReinterpretCastOp>(
        loc, memRefType,
        funcOp.getArgument(variablesOffset + accessedVar.index()),
        stridesAndOffset.second, memRefType.getShape(), stridesAndOffset.first);

    auto toTensorOp = builder.create<mlir::bufferization::ToTensorOp>(
        loc, tensorType, reinterpretOp);

    toTensorOp.setRestrict(true);

    variables[variableOp.getSymName()] = toTensorOp;
  }

  // Clone the original operations.
  for (auto &nestedOp : equationOp.getTemplate().getOps()) {
    if (mlir::isa<EquationSideOp, EquationSidesOp>(nestedOp)) {
      continue;
    }

    if (auto timeOp = mlir::dyn_cast<TimeOp>(nestedOp)) {
      auto castOp = builder.create<CastOp>(
          loc, RealType::get(builder.getContext()), funcOp.getArgument(0));
      mapping.map(timeOp, castOp.getResult());
      continue;
    }

    if (auto getOp = mlir::dyn_cast<VariableGetOp>(nestedOp)) {
      assert(variables.contains(getOp.getVariable()));
      mlir::Value mappedValue = variables.lookup(getOp.getVariable());

      if (auto tensorType = mlir::dyn_cast<mlir::TensorType>(getOp.getType())) {
        if (tensorType != mappedValue.getType()) {
          mappedValue = builder
                            .create<mlir::UnrealizedConversionCastOp>(
                                loc, tensorType, mappedValue)
                            .getResult(0);
        }
      } else {
        mappedValue = builder.create<TensorExtractOp>(loc, mappedValue);
        mappedValue = builder.create<CastOp>(loc, getOp.getType(), mappedValue);
      }

      mapping.map(getOp, mappedValue);
      continue;
    }

    builder.clone(nestedOp, mapping);
  }

  mlir::Value returnValue =
      mapping.lookup(equationOp.getTemplate().getValueAtPath(path));

  returnValue = builder.create<CastOp>(loc, builder.getF64Type(), returnValue);
  builder.create<mlir::func::ReturnOp>(loc, returnValue);

  return funcOp;
}

mlir::LogicalResult createEquationFunctions(
    mlir::ModuleOp moduleOp, mlir::SymbolTableCollection &symbolTableCollection,
    ModelOp modelOp, EquationInstanceOp equationOp,
    llvm::StringRef functionBaseName, EquationFunctionsMap &mapping) {
  // Collect the accessed variables.
  llvm::SmallVector<VariableOp> accessedVariables;
  llvm::DenseSet<VariableOp> uniqueAccessedVariables;

  equationOp.getTemplate().walk([&](VariableGetOp getOp) {
    auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, getOp.getVariableAttr());

    if (!uniqueAccessedVariables.contains(variableOp)) {
      uniqueAccessedVariables.insert(variableOp);
      accessedVariables.push_back(variableOp);
    }
  });

  // Create the functions.
  mlir::func::FuncOp lhsFunc = createEquationSideFunction(
      moduleOp, symbolTableCollection, functionBaseName.str() + "_lhs",
      equationOp, accessedVariables, EquationPath(EquationPath::LEFT, 0));

  mlir::func::FuncOp rhsFunc = createEquationSideFunction(
      moduleOp, symbolTableCollection, functionBaseName.str() + "_rhs",
      equationOp, accessedVariables, EquationPath(EquationPath::RIGHT, 0));

  mapping[equationOp] = {lhsFunc, rhsFunc, accessedVariables};

  return mlir::success();
}

mlir::LogicalResult
createEquationFunctions(mlir::ModuleOp moduleOp, ModelOp modelOp,
                        llvm::ArrayRef<EquationInstanceOp> initialEquations,
                        llvm::ArrayRef<EquationInstanceOp> dynamicEquations,
                        EquationFunctionsMap &mapping) {
  mlir::SymbolTableCollection symbolTableCollection;

  uint64_t initialEquationsCounter = 0;
  uint64_t dynamicEquationsCounter = 0;

  for (EquationInstanceOp equationOp : initialEquations) {
    if (mlir::failed(createEquationFunctions(
            moduleOp, symbolTableCollection, modelOp, equationOp,
            "initial_equation_" + std::to_string(initialEquationsCounter++),
            mapping))) {
      return mlir::failure();
    }
  }

  for (EquationInstanceOp equationOp : dynamicEquations) {
    if (mlir::failed(createEquationFunctions(
            moduleOp, symbolTableCollection, modelOp, equationOp,
            "equation_" + std::to_string(dynamicEquationsCounter++),
            mapping))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

bool hasHelpFlag(int argc, char *argv[]) {
  for (int i = 0; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);

    if (arg == "--help") {
      return true;
    }
  }

  return false;
}

double getTolerance(int argc, char *argv[]) {
  double result = TOLERANCE_DEFAULT;

  for (int i = 0; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);

    if (arg.starts_with("--tolerance=")) {
      result = std::stod(arg.substr(12).str());
    }
  }

  return result;
}

void getLinkedLibs(int argc, char *argv[],
                   llvm::SmallVectorImpl<llvm::StringRef> &libs) {
  for (int i = 0; i < argc - 1; ++i) {
    if (strcmp(argv[i], "-l") == 0) {
      libs.push_back(argv[i + 1]);
      ++i;
    }
  }
}

size_t getVariableFlatIndex(llvm::ArrayRef<int64_t> shape,
                            llvm::ArrayRef<int64_t> indices) {
  assert(shape.size() == indices.size());
  size_t offset = indices[0];

  for (size_t i = 1, e = shape.size(); i < e; ++i) {
    offset = offset * shape[i] + indices[i];
  }

  return offset;
}

std::pair<mlir::SymbolRefAttr, uint64_t>
getBaseVariable(VariableOp variableOp, const DerivativesMap &derivativesMap) {
  mlir::SymbolRefAttr baseVar =
      mlir::SymbolRefAttr::get(variableOp.getSymNameAttr());
  uint64_t order = 0;

  std::optional<mlir::SymbolRefAttr> derivedVar =
      derivativesMap.getDerivedVariable(baseVar);

  while (derivedVar) {
    ++order;
    baseVar = *derivedVar;
    derivedVar = derivativesMap.getDerivedVariable(baseVar);
  }

  return {baseVar, order};
}

bool setVariableValues(
    VariableOp variableOp, const DerivativesMap &derivativesMap, double *data,
    llvm::ArrayRef<std::pair<std::string, double>> variables) {
  bool found = false;

  for (const auto &variable : variables) {
    std::string regexStr = "^";

    mlir::SymbolRefAttr baseVar;
    uint64_t derOrder = 0;
    std::tie(baseVar, derOrder) = getBaseVariable(variableOp, derivativesMap);

    std::string baseVarName = baseVar.getRootReference().str();

    for (mlir::FlatSymbolRefAttr nestedRef : baseVar.getNestedReferences()) {
      baseVarName += "." + nestedRef.getValue().str();
    }

    for (uint64_t order = 0; order < derOrder; ++order) {
      regexStr += "der\\(";
    }

    regexStr += llvm::Regex::escape(baseVarName);

    if (!variableOp.getVariableType().isScalar()) {
      regexStr += "\\[([0-9]+(,[0-9]+)*)\\]";
    }

    for (uint64_t order = 0; order < derOrder; ++order) {
      regexStr += "\\)";
    }

    regexStr += "$";

    llvm::Regex regex(regexStr);
    llvm::SmallVector<llvm::StringRef, 4> matches;

    if (regex.match(variable.first, &matches)) {
      found = true;
      size_t flatIndex = 0;

      if (matches.size() > 1) {
        std::stringstream ss(matches[1].str());
        std::string index;
        std::vector<int64_t> indices;

        while (std::getline(ss, index, ',')) {
          indices.push_back(std::stoi(index) - 1);
        }

        flatIndex = getVariableFlatIndex(
            variableOp.getVariableType().getShape(), indices);
      }

      data[flatIndex] = variable.second;
    }
  }

  return found;
}

bool checkEquations(
    mlir::ExecutionEngine &executionEngine, ModelOp modelOp,
    llvm::ArrayRef<VariableOp> variableOps,
    llvm::ArrayRef<EquationInstanceOp> equations,
    const EquationFunctionsMap &equationFunctions,
    llvm::DenseMap<VariableOp, StridedMemRefType<double, 1>> &variableMemRefs,
    llvm::ArrayRef<std::pair<std::string, double>> variables,
    double tolerance) {
  bool result = true;

  // Set the content of the variable memrefs.
  for (VariableOp variableOp : variableOps) {
    auto memRefIt = variableMemRefs.find(variableOp);
    assert(memRefIt != variableMemRefs.end());
    double *data = memRefIt->getSecond().data;

    if (!setVariableValues(variableOp, modelOp.getProperties().derivativesMap,
                           data, variables)) {
      llvm::errs() << "No data found for variable " << variableOp.getSymName()
                   << "\n";

      return false;
    }
  }

  for (EquationInstanceOp equationOp : equations) {
    const auto &functions = equationFunctions.lookup(equationOp);
    llvm::SmallVector<void *> args;

    // Time value.
    auto timeIt = llvm::find_if(
        variables, [](const std::pair<std::string, double> &variable) {
          return variable.first == "time";
        });

    assert(timeIt != variables.end() && "Time variable not found");
    double timeValue = timeIt->second;
    mlir::ExecutionEngine::Argument<double>::pack(args, timeValue);

    // Induction values.
    llvm::SmallVector<int64_t> inductions;
    inductions.resize(equationOp.getInductionVariables().size());

    for (int64_t &induction : inductions) {
      mlir::ExecutionEngine::Argument<int64_t>::pack(args, induction);
    }

    // Variable memrefs.
    llvm::SmallVector<UnrankedMemRefType<double>> unrankedMemRefs;

    for (VariableOp accessedVariable : std::get<2>(functions)) {
      auto memRefIt = variableMemRefs.find(accessedVariable);
      assert(memRefIt != variableMemRefs.end() && "Variable not found");

      unrankedMemRefs.push_back(
          {.rank = 1, .descriptor = &memRefIt->getSecond()});
    }

    llvm::SmallVector<void *> unrankedMemRefPtrs;

    for (auto &unrankedMemRef : unrankedMemRefs) {
      mlir::ExecutionEngine::Argument<decltype(unrankedMemRef)>::pack(
          unrankedMemRefPtrs, unrankedMemRef);
    }

    for (auto &unrankedMemRefPtr : unrankedMemRefPtrs) {
      args.push_back(&unrankedMemRefPtr);
    }

    // Function result.
    double functionSideResultValue;

    mlir::ExecutionEngine::Result<double> functionSideResult(
        functionSideResultValue);

    mlir::ExecutionEngine::Argument<decltype(functionSideResult)>::pack(
        args, functionSideResult);

    auto checkSidesFn =
        [&](std::optional<Point> indices) -> std::optional<bool> {
      // Set the current indices.
      if (indices) {
        for (size_t dim = 0, rank = indices->rank(); dim < rank; ++dim) {
          inductions[dim] = (*indices)[dim];
        }
      }

      // Call the left-hand side function.
      mlir::func::FuncOp lhsFuncOp = std::get<0>(functions);
      functionSideResultValue = 0xDEADEEF;

      auto lhsCall = executionEngine.invokePacked(
          "_mlir_ciface_" + lhsFuncOp.getSymName().str(), args);

      double lhs = functionSideResultValue;

      if (lhsCall) {
        llvm::errs() << "Can't compute left-hand side value\n";
        llvm::errs() << "  - ";
        equationOp.printInline(llvm::errs());

        if (indices) {
          llvm::errs() << "\n" << "  - Indices: " << indices << "\n";
        }

        return std::nullopt;
      }

      // Call the right-hand side function.
      mlir::func::FuncOp rhsFuncOp = std::get<1>(functions);
      functionSideResultValue = 0xDEADBEEF;

      auto rhsCall = executionEngine.invokePacked(
          "_mlir_ciface_" + rhsFuncOp.getSymName().str(), args);

      double rhs = functionSideResultValue;

      if (rhsCall) {
        llvm::errs() << "Can't compute right-hand side value\n";
        llvm::errs() << "  - ";
        equationOp.printInline(llvm::errs());

        if (indices) {
          llvm::errs() << "\n" << "  - Indices: " << indices << "\n";
        }

        return std::nullopt;
      }

      // Check if the two sides have equal values.
      if (auto difference = std::abs(lhs - rhs); difference > tolerance) {
        llvm::outs() << "Equality doesn't hold\n";
        llvm::outs() << "  - ";
        equationOp.printInline(llvm::outs());
        llvm::outs() << "\n";

        if (indices) {
          llvm::outs() << "  - Indices: " << indices << "\n";
        }

        llvm::outs() << "  - Time: " << timeValue << "\n"
                     << "  - LHS: " << lhs << "\n"
                     << "  - RHS: " << rhs << "\n"
                     << "  - Difference: " << difference << "\n";

        result = false;
      }

      return true;
    };

    if (equationOp.getInductionVariables().empty()) {
      // Scalar equation.
      auto localResult = checkSidesFn(std::nullopt);

      if (!localResult) {
        return false;
      }

      result &= *localResult;
    } else {
      // Array equation.
      // Iterate on all the indices.

      for (Point indices : equationOp.getProperties().indices) {
        auto localResult = checkSidesFn(indices);

        if (!localResult) {
          return false;
        }

        result &= *localResult;
      }
    }
  }

  return result;
}
} // namespace marco::verifier
