#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "marco/codegen/dialects/modelica/ModelicaBuilder.h"
#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/codegen/passes/SolveModel.h"
#include "marco/codegen/passes/TypeConverter.h"
#include "marco/codegen/passes/model/Equation.h"
#include "marco/codegen/passes/model/EquationImpl.h"
#include "marco/codegen/passes/model/Matching.h"
#include "marco/codegen/passes/model/Model.h"
#include "marco/codegen/passes/model/Scheduling.h"
#include "marco/modeling/Cycles.h"
#include "marco/utils/VariableFilter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>
#include <memory>
#include <set>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

// TODO factor out from here and AD pass (and maybe also from somewhere else)
template <class T>
unsigned int numDigits(T number)
{
  unsigned int digits = 0;

  while (number != 0)
  {
    number /= 10;
    ++digits;
  }

  return digits;
}

// TODO factor out from here and AD pass
static std::string getFullDerVariableName(llvm::StringRef baseName, unsigned int order)
{
  assert(order > 0);

  if (order == 1)
    return "der_" + baseName.str();

  return "der_" + std::to_string(order) + "_" + baseName.str();
}

// TODO factor out from here and AD pass
static std::string getNextFullDerVariableName(llvm::StringRef currentName, unsigned int requestedOrder)
{
  if (requestedOrder == 1)
    return getFullDerVariableName(currentName, requestedOrder);

  assert(currentName.rfind("der_") == 0);

  if (requestedOrder == 2)
    return getFullDerVariableName(currentName.substr(4), requestedOrder);

  return getFullDerVariableName(currentName.substr(5 + numDigits(requestedOrder - 1)), requestedOrder);
}

class ModelConverter
{
  private:
    // The derivatives map keeps track of whether a variable is the derivative
    // of another one. Each variable is identified by its position within the
    // list of the "body" region arguments.

    using DerivativesPositionsMap = std::map<size_t, size_t>;

    // Name for the functions of the simulation
    static constexpr llvm::StringLiteral mainFunctionName = "main";
    static constexpr llvm::StringLiteral initFunctionName = "init";
    static constexpr llvm::StringLiteral stepFunctionName = "step";
    static constexpr llvm::StringLiteral updateStateVariablesFunctionName = "updateStateVariables";
    static constexpr llvm::StringLiteral printHeaderFunctionName = "printHeader";
    static constexpr llvm::StringLiteral printFunctionName = "print";
    static constexpr llvm::StringLiteral deinitFunctionName = "deinit";
    static constexpr llvm::StringLiteral runFunctionName = "runSimulation";

  public:
    ModelConverter(
        SolveModelOptions options,
        TypeConverter& typeConverter)
      : options(std::move(options)),
        typeConverter(&typeConverter)
    {
    }

    mlir::LogicalResult convert(
        mlir::OpBuilder& builder,
        const Model<ScheduledEquation>& model,
        const mlir::BlockAndValueMapping& derivatives) const
    {
      ModelOp modelOp = model.getOperation();

      // Convert the original derivatives map between values into a map between positions
      DerivativesPositionsMap derivativesPositions;

      for (size_t i = 0, e = modelOp.body().getNumArguments(); i < e; ++i) {
        mlir::Value var = modelOp.body().getArgument(i);

        if (derivatives.contains(var)) {
          mlir::Value derivative = derivatives.lookup(var);
          bool derivativeFound = false;
          unsigned int position = 0;

          for (size_t j = 0; j < e && !derivativeFound; ++j) {
            mlir::Value arg = modelOp.body().getArgument(j);

            if (arg == derivative) {
              derivativeFound = true;
              position = j;
            }
          }

          assert(derivativeFound && "Derivative not found among arguments");
          derivativesPositions[i] = position;
        }
      }

      // Create the various functions composing the simulation
      if (auto res = createInitFunction(builder, modelOp); failed(res)) {
        model.getOperation().emitError("Could not create the '" + initFunctionName + "' function");
        return res;
      }

      if (auto res = createDeinitFunction(builder, modelOp); failed(res)) {
        model.getOperation().emitError("Could not create the '" + deinitFunctionName + "' function");
        return res;
      }

      if (auto res = createStepFunction(builder, model, derivatives); mlir::failed(res)) {
        model.getOperation().emitError("Could not create the '" + stepFunctionName + "' function");
        return res;
      }

      if (auto res = createUpdateStateVariablesFunction(builder, modelOp, derivativesPositions); mlir::failed(res)) {
        model.getOperation().emitError("Could not create the '" + updateStateVariablesFunctionName + "' function");
        return res;
      }

      if (auto res = createPrintHeaderFunction(builder, modelOp, derivativesPositions); failed(res)) {
        model.getOperation().emitError("Could not create the '" + printHeaderFunctionName + "' function");
        return res;
      }

      if (auto res = createPrintFunction(builder, modelOp, derivativesPositions); failed(res)) {
        model.getOperation().emitError("Could not create the '" + printFunctionName + "' function");
        return res;
      }

      if (options.emitMain) {
        if (auto res = createMainFunction(builder, model); mlir::failed(res)) {
          model.getOperation().emitError("Could not create the '" + mainFunctionName + "' function");
          return res;
        }
      }

      // Erase the model operation, which has been converted to algorithmic code
      model.getOperation().erase();

      return mlir::success();
    }

  private:
    /// Get the MLIR type corresponding to void*.
    mlir::Type getVoidPtrType() const
    {
      return mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(&typeConverter->getContext(), 8));
    }

    /// Get the MLIR function with the given name, or declare it inside the module if not present.
    mlir::LLVM::LLVMFuncOp getOrInsertFunction(
        mlir::OpBuilder& builder,
        mlir::ModuleOp module,
        llvm::StringRef name,
        mlir::LLVM::LLVMFunctionType type) const
    {
      if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return foo;
      }

      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      return builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, type);
    }

    /// Create the main function, which is called when the executable of the simulation is run.
    /// In order to keep the code generation simpler, the real implementation of the function
    /// managing the simulation lives within the runtime library and the main just consists in
    /// a call to such function.
    template<typename EquationType>
    mlir::LogicalResult createMainFunction(mlir::OpBuilder& builder, const Model<EquationType>& model) const
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      ModelOp modelOp = model.getOperation();
      mlir::Location loc = modelOp.getLoc();

      // Create the function inside the parent module
      auto module = modelOp->getParentOfType<mlir::ModuleOp>();
      builder.setInsertionPointToEnd(module.getBody());

      llvm::SmallVector<mlir::Type, 3> argsTypes;
      llvm::SmallVector<mlir::Type, 3> resultsTypes;

      argsTypes.push_back(builder.getI32Type());
      argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMPointerType::get(builder.getIntegerType(8))));
      resultsTypes.push_back(builder.getI32Type());

      auto function = builder.create<mlir::FuncOp>(
          loc, mainFunctionName, builder.getFunctionType(argsTypes, resultsTypes));

      auto* entryBlock = function.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      mlir::Type voidType = mlir::LLVM::LLVMVoidType::get(modelOp.getContext());

      auto runFunction = getOrInsertFunction(
          builder, module, runFunctionName, mlir::LLVM::LLVMFunctionType::get(voidType, llvm::None));

      builder.create<mlir::LLVM::CallOp>(loc, runFunction, llvm::None);

      mlir::Value returnValue = builder.create<mlir::ConstantOp>(loc, builder.getI32IntegerAttr(0));
      builder.create<mlir::ReturnOp>(loc, returnValue);

      return mlir::success();
    }

    /// Load the data structure from the opaque pointer that is passed around the
    /// simulation functions.
    ///
    /// @param builder	 operation builder
    /// @param ptr 	     opaque pointer
    /// @param varTypes  types of the variables
    /// @return data structure containing the variables
    mlir::Value loadDataFromOpaquePtr(
        mlir::OpBuilder& builder,
        mlir::Value ptr,
        mlir::TypeRange varTypes) const
    {
      mlir::Location loc = ptr.getLoc();
      llvm::SmallVector<mlir::Type, 3> structTypes;

      for (const auto& type : varTypes) {
        structTypes.push_back(typeConverter->convertType(type));
      }

      mlir::Type structType = mlir::LLVM::LLVMStructType::getLiteral(ptr.getContext(), structTypes);
      mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(structType);
      mlir::Value structPtr = builder.create<mlir::LLVM::BitcastOp>(loc, structPtrType, ptr);
      mlir::Value structValue = builder.create<mlir::LLVM::LoadOp>(loc, structPtr);

      return structValue;
    }

    /// Extract a value from the data structure shared between the various
    /// simulation main functions.
    ///
    /// @param builder 			  operation builder
    /// @param typeConverter  type converter
    /// @param structValue 	  data structure
    /// @param type 				  value type
    /// @param position 		  value position
    /// @return extracted value
    mlir::Value extractValue(
        mlir::OpBuilder& builder,
        mlir::Value structValue,
        mlir::Type type,
        unsigned int position) const
    {
      mlir::Location loc = structValue.getLoc();

      assert(structValue.getType().isa<mlir::LLVM::LLVMStructType>() && "Not an LLVM struct");
      auto structType = structValue.getType().cast<mlir::LLVM::LLVMStructType>();
      auto structTypes = structType.getBody();
      assert (position < structTypes.size() && "LLVM struct: index is out of bounds");

      mlir::Value var = builder.create<mlir::LLVM::ExtractValueOp>(loc, structTypes[position], structValue, builder.getIndexArrayAttr(position));
      return typeConverter->materializeSourceConversion(builder, loc, type, var);
    }

    /// Bufferize the variables and convert the subsequent load/store operations to operate on the
    /// allocated memory buffer.
    mlir::Value convertMember(mlir::OpBuilder& builder, MemberCreateOp op) const
    {
      mlir::OpBuilder::InsertionGuard guard(builder);

      using LoadReplacer = std::function<void(MemberLoadOp)>;
      using StoreReplacer = std::function<void(MemberStoreOp)>;

      mlir::Location loc = op->getLoc();

      auto memberType = op.resultType().cast<MemberType>();
      auto arrayType = memberType.toArrayType();
      assert(arrayType.getAllocationScope() == BufferAllocationScope::heap);

      // Create the memory buffer for the variable
      builder.setInsertionPoint(op);

      mlir::Value reference = builder.create<AllocOp>(
          loc, arrayType.getElementType(), arrayType.getShape(), op.dynamicDimensions(), false);

      // Replace loads and stores with appropriate instructions operating on the new memory buffer.
      // The way such replacements are executed depend on the nature of the variable.

      auto replacers = [&]() {
        if (arrayType.isScalar()) {
          assert(op.dynamicDimensions().empty());

          auto loadReplacer = [&builder, reference](MemberLoadOp loadOp) -> void {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(loadOp);
            loadOp.replaceAllUsesWith(reference);
            loadOp.erase();
          };

          auto storeReplacer = [&builder, reference](MemberStoreOp storeOp) -> void {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(storeOp);
            auto assignment = builder.create<AssignmentOp>(storeOp.getLoc(), storeOp.value(), reference);
            storeOp->replaceAllUsesWith(assignment);
            storeOp.erase();
          };

          return std::make_pair<LoadReplacer, StoreReplacer>(loadReplacer, storeReplacer);
        }

        auto loadReplacer = [&builder, reference](MemberLoadOp loadOp) -> void {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(loadOp);
          loadOp.replaceAllUsesWith(reference);
          loadOp.erase();
        };

        auto storeReplacer = [&builder, reference](MemberStoreOp storeOp) -> void {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(storeOp);
          auto assignment = builder.create<AssignmentOp>(storeOp.getLoc(), storeOp.value(), reference);
          storeOp->replaceAllUsesWith(assignment);
          storeOp.erase();
        };

        return std::make_pair<LoadReplacer, StoreReplacer>(loadReplacer, storeReplacer);
      };

      LoadReplacer loadReplacer;
      StoreReplacer storeReplacer;
      std::tie(loadReplacer, storeReplacer) = replacers();

      for (auto* user : op->getUsers()) {
        if (auto loadOp = mlir::dyn_cast<MemberLoadOp>(user)) {
          loadReplacer(loadOp);
        } else if (auto storeOp = mlir::dyn_cast<MemberStoreOp>(user)) {
          storeReplacer(storeOp);
        }
      }

      op.replaceAllUsesWith(reference);
      op.erase();
      return reference;
    }

    mlir::LLVM::LLVMFuncOp lookupOrCreateHeapAllocFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
    {
      std::string name = "_MheapAlloc_pvoid_i64";

      if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return foo;
      }

      mlir::PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(getVoidPtrType(), builder.getI64Type());
      return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
    }

    /// Create the initialization function that allocates the variables and
    /// stores them into an appropriate data structure to be passed to the other
    /// simulation functions.
    mlir::LogicalResult createInitFunction(
        mlir::OpBuilder& builder, ModelOp modelOp) const
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Location loc = modelOp.getLoc();
      auto module = modelOp->getParentOfType<mlir::ModuleOp>();

      // Create the function inside the parent module
      builder.setInsertionPointToEnd(module.getBody());

      auto functionType = builder.getFunctionType(llvm::None, getVoidPtrType());
      auto function = builder.create<mlir::FuncOp>(loc, initFunctionName, functionType);

      mlir::BlockAndValueMapping mapping;

      // Move the initialization instructions into the new function
      modelOp.init().cloneInto(&function.getBody(), mapping);

      llvm::SmallVector<mlir::Value, 3> values;
      auto terminator = mlir::cast<YieldOp>(function.getBody().back().getTerminator());
      builder.setInsertionPointAfter(terminator);

      auto removeAllocationScopeFn = [&](mlir::Value value) -> mlir::Value {
        return builder.create<ArrayCastOp>(
            loc, value,
            value.getType().cast<ArrayType>().toUnknownAllocationScope());
      };

      // Add variables to the struct to be passed around (i.e. to the step and
      // print functions).

      for (const auto& var : terminator.values()) {
        auto memberCreateOp = var.getDefiningOp<MemberCreateOp>();
        mlir::Value array = convertMember(builder, memberCreateOp);
        builder.setInsertionPointAfterValue(array);
        values.push_back(removeAllocationScopeFn(array));
      }

      builder.setInsertionPointAfter(terminator);

      // Set the start time
      mlir::Value startTime = builder.create<ConstantOp>(loc, modelOp.startTime());
      builder.create<StoreOp>(loc, startTime, values[0]);

      // Pack the values
      mlir::TypeRange varTypes = modelOp.body().getArgumentTypes();
      std::vector<mlir::Type> structTypes;

      for (const auto& type : varTypes) {
        structTypes.push_back(typeConverter->convertType(type));
      }

      auto structType = mlir::LLVM::LLVMStructType::getLiteral(modelOp.getContext(), structTypes);
      mlir::Value structValue = builder.create<mlir::LLVM::UndefOp>(loc, structType);

      for (const auto& var : llvm::enumerate(values)) {
        mlir::Type convertedType = typeConverter->convertType(var.value().getType());
        mlir::Value convertedVar = typeConverter->materializeTargetConversion(builder, loc, convertedType, var.value());
        structValue = builder.create<mlir::LLVM::InsertValueOp>(loc, structValue, convertedVar, builder.getIndexArrayAttr(var.index()));
      }

      // The data structure must be stored on the heap in order to escape
      // from the function.

      // Add the "malloc" function to the module
      auto heapAllocFunc = lookupOrCreateHeapAllocFn(builder, module);

      // Determine the size (in bytes) of the memory to be allocated
      mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(structType);
      mlir::Value nullPtr = builder.create<mlir::LLVM::NullOp>(loc, structPtrType);

      mlir::Value one = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));
      one = typeConverter->materializeTargetConversion(builder, loc, typeConverter->getIndexType(), one);

      mlir::Value gepPtr = builder.create<mlir::LLVM::GEPOp>(loc, structPtrType, llvm::ArrayRef<mlir::Value>{nullPtr, one});
      mlir::Value sizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(loc, typeConverter->getIndexType(), gepPtr);
      mlir::Value resultOpaquePtr = createLLVMCall(builder, loc, heapAllocFunc, sizeBytes, getVoidPtrType())[0];

      // Store the struct into the heap memory
      mlir::Value resultCastedPtr = builder.create<mlir::LLVM::BitcastOp>(loc, structPtrType, resultOpaquePtr);
      builder.create<mlir::LLVM::StoreOp>(loc, structValue, resultCastedPtr);

      builder.create<mlir::ReturnOp>(loc, resultOpaquePtr);
      terminator->erase();

      return mlir::success();
    }

    mlir::LLVM::LLVMFuncOp lookupOrCreateHeapFreeFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
    {
      std::string name = "_MheapFree_void_pvoid";

      if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return foo;
      }

      mlir::PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      mlir::Type voidType = mlir::LLVM::LLVMVoidType::get(module.getContext());
      auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(voidType, getVoidPtrType());
      return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
    }

    /// Create a function to be called when the simulation has finished and the
    /// variables together with its data structure are not required anymore and
    /// thus can be deallocated.
    mlir::LogicalResult createDeinitFunction(mlir::OpBuilder& builder, ModelOp modelOp) const
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Location loc = modelOp.getLoc();
      auto module = modelOp->getParentOfType<mlir::ModuleOp>();

      // Create the function inside the parent module
      builder.setInsertionPointToEnd(module.getBody());

      auto function = builder.create<mlir::FuncOp>(
          loc, deinitFunctionName,
          builder.getFunctionType(getVoidPtrType(), llvm::None));

      auto* entryBlock = function.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      // Extract the data from the struct
      mlir::TypeRange varTypes = modelOp.body().getArgumentTypes();
      mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), varTypes);

      // Deallocate the arrays
      for (const auto& type : llvm::enumerate(varTypes)) {
        if (auto arrayType = type.value().dyn_cast<ArrayType>()) {
          mlir::Value var = extractValue(builder, structValue, varTypes[type.index()], type.index());
          var = builder.create<ArrayCastOp>(loc, var, arrayType.toAllocationScope(BufferAllocationScope::heap));
          builder.create<FreeOp>(loc, var);
        }
      }

      // Add "free" function to the module
      auto freeFunc = lookupOrCreateHeapFreeFn(builder, module);

      // Deallocate the data structure
      builder.create<mlir::LLVM::CallOp>(
          loc, llvm::None, builder.getSymbolRefAttr(freeFunc), function.getArgument(0));

      builder.create<mlir::ReturnOp>(loc);
      return mlir::success();
    }

    mlir::FuncOp createEquationFunction(
        mlir::OpBuilder& builder,
        const ScheduledEquation& equation,
        llvm::StringRef equationFunctionName,
        mlir::FuncOp templateFunction,
        mlir::TypeRange varsTypes) const
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Location loc = equation.getOperation().getLoc();

      auto module = equation.getOperation()->getParentOfType<mlir::ModuleOp>();
      builder.setInsertionPointToEnd(module.getBody());

      auto functionType = builder.getFunctionType(varsTypes, llvm::None);
      auto function = builder.create<mlir::FuncOp>(loc, equationFunctionName, functionType);

      auto* entryBlock = function.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      auto valuesFn = [&](marco::modeling::scheduling::Direction iterationDirection, size_t index) -> std::tuple<mlir::Value, mlir::Value, mlir::Value> {
        assert(iterationDirection == marco::modeling::scheduling::Direction::Forward ||
            iterationDirection == marco::modeling::scheduling::Direction::Backward);

        if (iterationDirection == marco::modeling::scheduling::Direction::Forward) {
          mlir::Value begin = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(equation.getRangeBegin(index)));
          mlir::Value end = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(equation.getRangeEnd(index)));
          mlir::Value step = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));

          return std::make_tuple(begin, end, step);
        }

        mlir::Value begin = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(equation.getRangeEnd(index) - 1));
        mlir::Value end = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(equation.getRangeBegin(index) - 1));
        mlir::Value step = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));

        return std::make_tuple(begin, end, step);
      };

      std::vector<mlir::Value> args;

      for (size_t i = 0, e = equation.getNumOfIterationVars(); i < e; ++i) {
        auto values = valuesFn(equation.getSchedulingDirection(), i);

        args.push_back(std::get<0>(values));
        args.push_back(std::get<1>(values));
        args.push_back(std::get<2>(values));
      }

      mlir::ValueRange vars = function.getArguments();
      args.insert(args.end(), vars.begin(), vars.end());

      // Call the equation template function
      builder.create<mlir::CallOp>(loc, templateFunction, args);

      builder.create<mlir::ReturnOp>(loc);
      return function;
    }

    /// Create the function to be called at each time step.
    mlir::LogicalResult createStepFunction(
        mlir::OpBuilder& builder,
        const Model<ScheduledEquation>& model,
        const mlir::BlockAndValueMapping& derivatives) const
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      ModelOp modelOp = model.getOperation();
      mlir::Location loc = modelOp.getLoc();

      // Create the function inside the parent module
      builder.setInsertionPointToEnd(modelOp->getParentOfType<mlir::ModuleOp>().getBody());

      auto function = builder.create<mlir::FuncOp>(
          loc, stepFunctionName,
          builder.getFunctionType(getVoidPtrType(), builder.getI1Type()));

      auto* entryBlock = function.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      // Extract the data from the struct
      mlir::TypeRange varTypes = modelOp.body().getArgumentTypes();
      mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), varTypes);

      llvm::SmallVector<mlir::Value, 3> vars;

      for (const auto& varType : llvm::enumerate(varTypes)) {
        vars.push_back(extractValue(builder, structValue, varTypes[varType.index()], varType.index()));
      }

      // Increment the time
      mlir::Value timeStep = builder.create<ConstantOp>(loc, modelOp.timeStep());
      mlir::Value currentTime = builder.create<LoadOp>(loc, vars[0]);
      mlir::Value increasedTime = builder.create<AddOp>(loc, currentTime.getType(), currentTime, timeStep);
      builder.create<StoreOp>(loc, increasedTime, vars[0]);

      // Check if the current time is less than the end time
      mlir::Value endTime = builder.create<ConstantOp>(loc, modelOp.endTime());

      mlir::Value condition = builder.create<LteOp>(loc, BooleanType::get(modelOp->getContext()), increasedTime, endTime);
      condition = typeConverter->materializeTargetConversion(builder, condition.getLoc(), builder.getI1Type(), condition);

      auto ifOp = builder.create<mlir::scf::IfOp>(loc, builder.getI1Type(), condition, true);
      builder.create<mlir::ReturnOp>(loc, ifOp.getResult(0));

      // Convert the equations into algorithmic code
      builder.setInsertionPointToStart(&ifOp.thenRegion().front());

      size_t equationTemplateCounter = 0;
      size_t equationCounter = 0;

      std::vector<mlir::FuncOp> equationTemplateFunctions;

      for (const auto& equation : model.getEquations()) {
        std::string templateFunctionName = "eq_template_" + std::to_string(equationTemplateCounter);
        ++equationTemplateCounter;

        auto clonedExplicitEquation = equation->cloneAndExplicitate(builder);

        if (clonedExplicitEquation == nullptr) {
          model.getOperation().emitError("Could not explicitate equation");
          return mlir::failure();
        }

        auto templateFunction = clonedExplicitEquation->createTemplateFunction(
            builder, templateFunctionName, modelOp.body().getArguments(), equation->getSchedulingDirection());

        equationTemplateFunctions.push_back(templateFunction);

        // Erase the temporary EquationOp clone (and its ForEquationOp parents) that has been used to
        // create the template function.
        clonedExplicitEquation->eraseIR();

        // Create the function that calls the template.
        // This function dictates the indices the template will work with.
        std::string equationFunctionName = "eq_" + std::to_string(equationCounter);
        ++equationCounter;

        auto equationFunction = createEquationFunction(
            builder, *equation, equationFunctionName, templateFunction, modelOp.body().getArgumentTypes());

        // Create the call to the instantiated template function
        builder.create<mlir::CallOp>(loc, equationFunction, vars);
      }

      // If we didn't reach the end time update the variables and return
      // true to continue the simulation.
      auto trueValue = builder.create<mlir::ConstantOp>(loc, builder.getBoolAttr(true));
      builder.create<mlir::scf::YieldOp>(loc, trueValue.getResult());

      // Otherwise, return false to stop the simulation
      builder.setInsertionPointToStart(&ifOp.elseRegion().front());
      mlir::Value falseValue = builder.create<mlir::ConstantOp>(loc, builder.getBoolAttr(false));
      builder.create<mlir::scf::YieldOp>(loc, falseValue);

      return mlir::success();
    }

    /// Create the functions that calculates the values that the state variables will have
    /// in the next iteration.
    mlir::LogicalResult createUpdateStateVariablesFunction(
        mlir::OpBuilder& builder, ModelOp modelOp, const DerivativesPositionsMap& derivatives) const
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Location loc = modelOp.getLoc();

      // Create the function inside the parent module
      builder.setInsertionPointToEnd(modelOp->getParentOfType<mlir::ModuleOp>().getBody());

      auto function = builder.create<mlir::FuncOp>(
          loc, updateStateVariablesFunctionName,
          builder.getFunctionType(getVoidPtrType(), llvm::None));

      auto* entryBlock = function.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      // Extract the state variables from the opaque pointer
      mlir::TypeRange varTypes = modelOp.body().getArgumentTypes();
      mlir::Value structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), varTypes);

      // Update the state variables by applying the forward Euler method
      mlir::Value timeStep = builder.create<ConstantOp>(loc, modelOp.timeStep());

      std::vector<std::pair<mlir::Value, mlir::Value>> varsAndDers;

      for (const auto& variable : modelOp.body().getArguments()) {
        size_t index = variable.getArgNumber();
        auto it = derivatives.find(variable.getArgNumber());

        if (it != derivatives.end()) {
          mlir::Value var = extractValue(builder, structValue, varTypes[index], index);
          mlir::Value der = extractValue(builder, structValue, varTypes[it->second], it->second);
          varsAndDers.emplace_back(var, der);
        }
      }

      for (const auto& [var, der] : varsAndDers) {
        mlir::Value nextValue = builder.create<MulOp>(loc, der.getType(), der, timeStep);
        nextValue = builder.create<AddOp>(loc, var.getType(), nextValue, var);
        builder.create<AssignmentOp>(loc, nextValue, var);
      }

      builder.create<mlir::ReturnOp>(loc);
      return mlir::success();
    }

    void printSeparator(mlir::OpBuilder& builder, mlir::Value separator) const
    {
      auto module = separator.getParentRegion()->getParentOfType<mlir::ModuleOp>();
      auto printfRef = getOrInsertPrintf(builder, module);
      builder.create<mlir::LLVM::CallOp>(separator.getLoc(), printfRef, separator);
    }

    mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder& builder, mlir::StringRef name, mlir::StringRef value, mlir::ModuleOp module) const
    {
      // Create the global at the entry of the module
      mlir::LLVM::GlobalOp global;

      if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
        mlir::OpBuilder::InsertionGuard insertGuard(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto type = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(builder.getContext(), 8), value.size());
        global = builder.create<mlir::LLVM::GlobalOp>(loc, type, true, mlir::LLVM::Linkage::Internal, name, builder.getStringAttr(value));
      }

      // Get the pointer to the first character in the global string
      mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);

      mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
          loc,
          mlir::IntegerType::get(builder.getContext(), 64),
          builder.getIntegerAttr(builder.getIndexType(), 0));

      return builder.create<mlir::LLVM::GEPOp>(
          loc,
          mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8)),
          globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
    }

    mlir::Value getSeparatorString(mlir::Location loc, mlir::OpBuilder& builder, mlir::ModuleOp module) const
    {
      return getOrCreateGlobalString(loc, builder, "semicolon", mlir::StringRef(";\0", 2), module);
    }

    mlir::Value getNewlineString(mlir::Location loc, mlir::OpBuilder& builder, mlir::ModuleOp module) const
    {
      return getOrCreateGlobalString(loc, builder, "newline", mlir::StringRef("\n\0", 2), module);
    }

    mlir::LLVM::LLVMFuncOp getOrInsertPrintf(mlir::OpBuilder& builder, mlir::ModuleOp module) const
    {
      auto *context = module.getContext();

      // Create a function declaration for printf, the signature is:
      //   * `i32 (i8*, ...)`
      auto llvmI32Ty = mlir::IntegerType::get(context, 32);
      auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
      auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

      // Insert the printf function into the body of the parent module
      return getOrInsertFunction(builder, module, "printf", llvmFnType);
    }

    void printVariableName(
        mlir::OpBuilder& builder,
        mlir::Value name,
        mlir::Type type,
        VariableFilter::Filter filter,
        std::function<mlir::Value()> structValue,
        unsigned int position,
        mlir::ModuleOp module,
        mlir::Value separator,
        bool shouldPreprendSeparator = true) const
    {
      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        if (arrayType.getRank() == 0) {
          printScalarVariableName(builder, name, module, separator, shouldPreprendSeparator);
        } else {
          printArrayVariableName(
              builder, name, type, filter, structValue, position, module, separator, shouldPreprendSeparator);
        }
      } else {
        printScalarVariableName(builder, name, module, separator, shouldPreprendSeparator);
      }
    }

    void printScalarVariableName(
        mlir::OpBuilder& builder,
        mlir::Value name,
        mlir::ModuleOp module,
        mlir::Value separator,
        bool shouldPrependSeparator) const
    {
      if (shouldPrependSeparator) {
        printSeparator(builder, separator);
      }

      mlir::Location loc = name.getLoc();
      mlir::Value formatSpecifier = getOrCreateGlobalString(loc, builder, "frmt_spec_str", mlir::StringRef("%s\0", 3), module);
      auto printfRef = getOrInsertPrintf(builder, module);
      builder.create<mlir::LLVM::CallOp>(loc, printfRef, mlir::ValueRange({ formatSpecifier, name }));
    }

    void printArrayVariableName(
        mlir::OpBuilder& builder,
        mlir::Value name,
        mlir::Type type,
        VariableFilter::Filter filter,
        std::function<mlir::Value()> structValue,
        unsigned int position,
        mlir::ModuleOp module,
        mlir::Value separator,
        bool shouldPrependSeparator) const
    {
      mlir::Location loc = name.getLoc();
      assert(type.isa<ArrayType>());

      // Get a reference to the printf function
      auto printfRef = getOrInsertPrintf(builder, module);

      // Create the brackets and comma strings
      mlir::Value lSquare = getOrCreateGlobalString(loc, builder, "lsquare", llvm::StringRef("[\0", 2), module);
      mlir::Value rSquare = getOrCreateGlobalString(loc, builder, "rsquare", llvm::StringRef("]\0", 2), module);
      mlir::Value comma = getOrCreateGlobalString(loc, builder, "comma", llvm::StringRef(",\0", 2), module);

      // Create the format strings
      mlir::Value stringFormatSpecifier = getOrCreateGlobalString(loc, builder, "frmt_spec_str", mlir::StringRef("%s\0", 3), module);
      mlir::Value integerFormatSpecifier = getOrCreateGlobalString(loc, builder, "frmt_spec_int", mlir::StringRef("%ld\0", 4), module);

      // Allow for the variable to lazily extracted if one of its dimension size
      // must be determined.
      bool valueLoaded = false;
      mlir::Value extractedValue = nullptr;
      auto insertionPoint = builder.saveInsertionPoint();

      auto var = [&]() -> mlir::Value {
        if (!valueLoaded) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.restoreInsertionPoint(insertionPoint);
          extractedValue = extractValue(builder, structValue(), type, position);
          valueLoaded = true;
        }

        return extractedValue;
      };

      // Create the lower and upper bounds
      auto ranges = filter.getRanges();
      auto arrayType = type.cast<ArrayType>();
      assert(arrayType.getRank() == ranges.size());

      llvm::SmallVector<mlir::Value, 3> lowerBounds;
      llvm::SmallVector<mlir::Value, 3> upperBounds;

      mlir::Value one = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));
      llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

      for (const auto& range : llvm::enumerate(ranges)) {
        // In Modelica, arrays are 1-based. If present, we need to lower by 1
        // the value given by the variable filter.

        unsigned int lowerBound = range.value().hasLowerBound() ? range.value().getLowerBound() - 1 : 0;
        lowerBounds.push_back(builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(lowerBound)));

        // The upper bound is not lowered because the SCF's for operation assumes
        // them as excluded.

        if (range.value().hasUpperBound()) {
          mlir::Value upperBound = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(range.value().getUpperBound()));
          upperBounds.push_back(upperBound);
        } else {
          mlir::Value dim = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(range.index()));
          mlir::Value upperBound = builder.create<DimOp>(loc, var(), dim);
          upperBounds.push_back(upperBound);
        }
      }

      bool shouldPrintSeparator = false;

      // Create nested loops in order to iterate on each dimension of the array
      mlir::scf::buildLoopNest(
          builder, loc, lowerBounds, upperBounds, steps,
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange indexes) {
            // Print the separator, the variable name and the left square bracket
            printSeparator(builder, separator);
            builder.create<mlir::LLVM::CallOp>(loc, printfRef, mlir::ValueRange({ stringFormatSpecifier, name }));
            builder.create<mlir::LLVM::CallOp>(loc, printfRef, lSquare);

            for (mlir::Value index : indexes) {
              if (shouldPrintSeparator)
                builder.create<mlir::LLVM::CallOp>(loc, printfRef, comma);

              shouldPrintSeparator = true;

              mlir::Type convertedType = typeConverter->convertType(index.getType());
              index = typeConverter->materializeTargetConversion(builder, loc, convertedType, index);

              // Arrays are 1-based in Modelica, so we add 1 in order to print
              // indexes that are coherent with the model source.
              mlir::Value increment = builder.create<mlir::ConstantOp>(loc, builder.getIntegerAttr(index.getType(), 1));
              index = builder.create<mlir::AddIOp>(loc, index.getType(), index, increment);

              builder.create<mlir::LLVM::CallOp>(loc, printfRef, mlir::ValueRange({ integerFormatSpecifier, index }));
            }

            // Print the right square bracket
            builder.create<mlir::LLVM::CallOp>(loc, printfRef, rSquare);
          });
    }

    mlir::LogicalResult createPrintHeaderFunction(
        mlir::OpBuilder& builder,
        ModelOp op,
        DerivativesPositionsMap& derivativesPositions) const
    {
      mlir::TypeRange varTypes = op.body().getArgumentTypes();

      auto callback = [&](std::function<mlir::Value()> structValue, llvm::StringRef name, unsigned int position, VariableFilter::Filter filter, mlir::Value separator) -> mlir::LogicalResult {
        mlir::Location loc = op.getLoc();
        auto module = op->getParentOfType<mlir::ModuleOp>();

        std::string symbolName = "var" + std::to_string(position);
        llvm::SmallString<10> terminatedName(name);
        terminatedName.append("\0");
        mlir::Value symbol = getOrCreateGlobalString(loc, builder, symbolName, llvm::StringRef(terminatedName.c_str(), terminatedName.size() + 1), module);

        bool shouldPrintSeparator = position != 0;
        printVariableName(builder, symbol, varTypes[position], filter, structValue, position, module, separator, shouldPrintSeparator);
        return mlir::success();
      };

      return createPrintFunctionBody(builder, op, varTypes, derivativesPositions, printHeaderFunctionName, callback);
    }

    void printVariable(mlir::OpBuilder& builder, mlir::Value var, VariableFilter::Filter filter, mlir::Value separator, bool shouldPreprendSeparator = true) const
    {
      if (auto arrayType = var.getType().dyn_cast<ArrayType>()) {
        if (arrayType.getRank() == 0) {
          mlir::Value value = builder.create<LoadOp>(var.getLoc(), var);
          printScalarVariable(builder, value, separator, shouldPreprendSeparator);
        } else {
          printArrayVariable(builder, var, filter, separator, shouldPreprendSeparator);
        }
      } else {
        printScalarVariable(builder, var, separator, shouldPreprendSeparator);
      }
    }

    void printScalarVariable(mlir::OpBuilder& builder, mlir::Value var, mlir::Value separator, bool shouldPreprendSeparator = true) const
    {
      if (shouldPreprendSeparator) {
        printSeparator(builder, separator);
      }

      printElement(builder, var);
    }

    void printArrayVariable(mlir::OpBuilder& builder, mlir::Value var, VariableFilter::Filter filter, mlir::Value separator, bool shouldPreprendSeparator = true) const
    {
      mlir::Location loc = var.getLoc();
      assert(var.getType().isa<ArrayType>());

      auto ranges = filter.getRanges();
      auto arrayType = var.getType().cast<ArrayType>();
      assert(arrayType.getRank() == ranges.size());

      llvm::SmallVector<mlir::Value, 3> lowerBounds;
      llvm::SmallVector<mlir::Value, 3> upperBounds;

      mlir::Value one = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));
      llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

      for (const auto& range : llvm::enumerate(ranges)) {
        // In Modelica, arrays are 1-based. If present, we need to lower by 1
        // the value given by the variable filter.

        unsigned int lowerBound = range.value().hasLowerBound() ? range.value().getLowerBound() - 1 : 0;
        lowerBounds.push_back(builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(lowerBound)));

        // The upper bound is not lowered because the SCF's for operation assumes
        // them as excluded.

        if (range.value().hasUpperBound()) {
          mlir::Value upperBound = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(range.value().getUpperBound()));
          upperBounds.push_back(upperBound);
        } else {
          mlir::Value dim = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(range.index()));
          mlir::Value upperBound = builder.create<DimOp>(loc, var, dim);
          upperBounds.push_back(upperBound);
        }
      }

      // Create nested loops in order to iterate on each dimension of the array
      mlir::scf::buildLoopNest(
          builder, loc, lowerBounds, upperBounds, steps,
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position) {
            mlir::Value value = nestedBuilder.create<LoadOp>(loc, var, position);

            printSeparator(nestedBuilder, separator);
            printElement(nestedBuilder, value);
          });
    }

    void printElement(mlir::OpBuilder& builder, mlir::Value value) const
    {
      mlir::Location loc = value.getLoc();
      auto module = value.getParentRegion()->getParentOfType<mlir::ModuleOp>();
      auto printfRef = getOrInsertPrintf(builder, module);

      mlir::Type convertedType = typeConverter->convertType(value.getType());
      value = typeConverter->materializeTargetConversion(builder, loc, convertedType, value);
      mlir::Type type = value.getType();

      mlir::Value formatSpecifier;

      if (type.isa<mlir::IntegerType>()) {
        formatSpecifier = getOrCreateGlobalString(
            loc, builder, "frmt_spec_int", mlir::StringRef("%ld\0", 4), module);
      } else if (type.isa<mlir::FloatType>()) {
        formatSpecifier = getOrCreateGlobalString(
            loc, builder, "frmt_spec_float", mlir::StringRef("%.12f\0", 6), module);
      } else {
        assert(false && "Unknown type");
      }

      builder.create<mlir::LLVM::CallOp>(value.getLoc(), printfRef, mlir::ValueRange({ formatSpecifier, value }));
    }

    mlir::LogicalResult createPrintFunction(
        mlir::OpBuilder& builder,
        ModelOp op,
        DerivativesPositionsMap& derivativesPositions) const
    {
      mlir::TypeRange varTypes = op.body().getArgumentTypes();

      auto callback = [&](std::function<mlir::Value()> structValue, llvm::StringRef name, unsigned int position, VariableFilter::Filter filter, mlir::Value separator) -> mlir::LogicalResult {
        mlir::Value var = extractValue(builder, structValue(), varTypes[position], position);
        bool shouldPrintSeparator = position != 0;
        printVariable(builder, var, filter, separator, shouldPrintSeparator);
        return mlir::success();
      };

      return createPrintFunctionBody(builder, op, varTypes, derivativesPositions, printFunctionName, callback);
    }

    mlir::LogicalResult createPrintFunctionBody(
        mlir::OpBuilder& builder,
        ModelOp op,
        mlir::TypeRange varTypes,
        DerivativesPositionsMap& derivativesPositions,
        llvm::StringRef functionName,
        std::function<mlir::LogicalResult(std::function<mlir::Value()>, llvm::StringRef, unsigned int, VariableFilter::Filter, mlir::Value)> elementCallback) const
    {
      mlir::Location loc = op.getLoc();
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto module = op->getParentOfType<mlir::ModuleOp>();

      // Create the function inside the parent module
      builder.setInsertionPointToEnd(module.getBody());

      auto function = builder.create<mlir::FuncOp>(
          loc, functionName,
          builder.getFunctionType(getVoidPtrType(), llvm::None));

      auto* entryBlock = function.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);

      // Create the separator and newline global strings
      mlir::Value separator = getSeparatorString(loc, builder, module);
      mlir::Value newline = getNewlineString(loc, builder, module);

      // Create the callback to load the data structure whenever needed
      bool structValueLoaded = false;
      mlir::Value structValue = nullptr;
      auto structValueInsertionPoint = builder.saveInsertionPoint();

      auto structValueCallback = [&]() -> mlir::Value {
        if (!structValueLoaded) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.restoreInsertionPoint(structValueInsertionPoint);
          structValue = loadDataFromOpaquePtr(builder, function.getArgument(0), varTypes);
        }

        return structValue;
      };

      // Get the names of the variables
      llvm::SmallVector<llvm::StringRef, 8> variableNames =
          llvm::to_vector<8>(op.variableNames().getAsValueRange<mlir::StringAttr>());

      // Map each variable to its position inside the data structure.
      // It must be noted that the data structure also contains derivative (if
      // existent), so its size can be greater than the number of names.

      assert(op.variableNames().size() <= varTypes.size());
      llvm::StringMap<size_t> variablePositionByName;

      for (const auto& var : llvm::enumerate(variableNames))
        variablePositionByName[var.value()] = var.index() + 1; // + 1 to skip the "time" variable

      // The positions have been saved, so we can now sort the names
      llvm::sort(variableNames, [](llvm::StringRef x, llvm::StringRef y) -> bool {
        return x.compare_insensitive(y) < 0;
      });

      if (auto status = elementCallback(
          structValueCallback, "time", 0, VariableFilter::Filter::visibleScalar(), separator); mlir::failed(status)) {
        return status;
      }

      // Print the other variables
      for (const auto& name : variableNames) {
        assert(variablePositionByName.count(name) != 0);
        size_t position = variablePositionByName[name];

        unsigned int rank = 0;

        if (auto arrayType = varTypes[position].dyn_cast<ArrayType>()) {
          rank = arrayType.getRank();
        }

        auto filter = options.variableFilter->getVariableInfo(name, rank);

        if (!filter.isVisible()) {
          continue;
        }

        if (auto status = elementCallback(
            structValueCallback, name, position, filter, separator); mlir::failed(status)) {
          return status;
        }
      }

      // Print the derivatives
      for (const auto& name : variableNames) {
        size_t varPosition = variablePositionByName[name];

        if (derivativesPositions.count(varPosition) == 0) {
          // The variable has no derivative
          continue;
        }

        size_t derivedVarPosition = derivativesPositions[varPosition];

        unsigned int rank = 0;

        if (auto arrayType = varTypes[derivedVarPosition].dyn_cast<ArrayType>()) {
          rank = arrayType.getRank();
        }

        auto filter = options.variableFilter->getVariableDerInfo(name, rank);

        if (!filter.isVisible()) {
          continue;
        }

        llvm::SmallString<15> derName;
        derName.append("der(");
        derName.append(name);
        derName.append(")");

        if (auto status = elementCallback(
            structValueCallback, derName, derivedVarPosition, filter, separator); mlir::failed(status)) {
          return status;
        }
      }

      // Print a newline character after all the variables have been processed
      builder.create<mlir::LLVM::CallOp>(loc, getOrInsertPrintf(builder, module), newline);

      builder.create<mlir::ReturnOp>(loc);
      return mlir::success();
    }

  private:
    SolveModelOptions options;
    TypeConverter* typeConverter;
};

/// Remove the derivative operations by replacing them with appropriate
/// buffers, and set the derived variables as state variables.
template<typename EquationType>
static mlir::LogicalResult removeDerivatives(
    mlir::OpBuilder& builder, Model<EquationType>& model, mlir::BlockAndValueMapping& derivatives)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  auto appendIndexesFn = [](llvm::SmallVectorImpl<mlir::Value>& destination, mlir::ValueRange indexes) {
    for (size_t i = 0, e = indexes.size(); i < e; ++i) {
      mlir::Value index = indexes[e - 1 - i];
      destination.push_back(index);
    }
  };

  auto derivativeOrder = [&](mlir::Value value) -> unsigned int {
    auto inverseMap = derivatives.getInverse();
    unsigned int result = 0;

    while (inverseMap.contains(value)) {
      ++result;
      value = inverseMap.lookup(value);
    }

    return result;
  };

  model.getOperation().walk([&](DerOp op) {
    mlir::Location loc = op->getLoc();
    mlir::Value operand = op.operand();

    // If the value to be derived belongs to an array, then also the derived
    // value is stored within an array. Thus, we need to store its position.

    llvm::SmallVector<mlir::Value, 3> subscriptions;

    while (!operand.isa<mlir::BlockArgument>())
    {
      mlir::Operation* definingOp = operand.getDefiningOp();
      assert(mlir::isa<LoadOp>(definingOp) || mlir::isa<SubscriptionOp>(definingOp));

      if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp))
      {
        appendIndexesFn(subscriptions, loadOp.indexes());
        operand = loadOp.memory();
      }

      auto subscriptionOp = mlir::cast<SubscriptionOp>(definingOp);
      appendIndexesFn(subscriptions, subscriptionOp.indexes());
      operand = subscriptionOp.source();
    }

    if (!derivatives.contains(operand))
    {
      auto model = op->getParentOfType<ModelOp>();
      auto terminator = mlir::cast<YieldOp>(model.init().back().getTerminator());
      builder.setInsertionPoint(terminator);

      size_t index = operand.cast<mlir::BlockArgument>().getArgNumber();
      auto memberCreateOp = terminator.values()[index].getDefiningOp<MemberCreateOp>();
      auto nextDerName = getNextFullDerVariableName(memberCreateOp.name(), derivativeOrder(operand) + 1);

      assert(operand.getType().isa<ArrayType>());
      auto arrayType = operand.getType().cast<ArrayType>().toElementType(RealType::get(builder.getContext()));

      // Create the member and initialize it
      auto memberType = MemberType::get(arrayType.toAllocationScope(BufferAllocationScope::heap));
      mlir::Value memberDer = builder.create<MemberCreateOp>(loc, nextDerName, memberType, memberCreateOp.dynamicDimensions(), memberCreateOp.isConstant());
      mlir::Value zero = builder.create<ConstantOp>(loc, RealAttribute::get(builder.getContext(), 0));
      mlir::Value array = builder.create<MemberLoadOp>(loc, memberType.unwrap(), memberDer);
      builder.create<FillOp>(loc, zero, array);

      // Update the terminator values
      llvm::SmallVector<mlir::Value, 3> args(terminator.values().begin(), terminator.values().end());
      args.push_back(memberDer);
      builder.create<YieldOp>(loc, args);
      terminator.erase();

      // Add the new argument to the body of the model
      auto bodyArgument = model.body().addArgument(arrayType);
      derivatives.map(operand, bodyArgument);
    }

    builder.setInsertionPoint(op);
    mlir::Value derVar = derivatives.lookup(operand);

    llvm::SmallVector<mlir::Value, 3> reverted(subscriptions.rbegin(), subscriptions.rend());

    if (!subscriptions.empty()) {
      derVar = builder.create<SubscriptionOp>(loc, derVar, reverted);
    }

    if (auto arrayType = derVar.getType().cast<ArrayType>(); arrayType.getRank() == 0) {
      derVar = builder.create<LoadOp>(loc, derVar);
    }

    op.replaceAllUsesWith(derVar);
    op.erase();
  });

  return mlir::success();
}

/// Get all the variables that are declared inside the Model operation, independently
/// from their nature (state variables, constants, etc.).
static Variables discoverVariables(ModelOp model)
{
  Variables result;

  mlir::ValueRange vars = model.body().getArguments();

  for (size_t i = 1; i < vars.size(); ++i) {
    result.add(std::make_unique<Variable>(vars[i]));
  }

  return result;
}

/// Get the equations that are declared inside the Model operation.
static Equations<Equation> discoverEquations(ModelOp model, const Variables& variables)
{
  Equations<Equation> result;

  model.walk([&](EquationOp equationOp) {
    result.add(Equation::build(equationOp, variables));
  });

  return result;
}

/// Match each scalar variable to a scalar equation.
template<typename EquationType>
static mlir::LogicalResult matching(
    Model<MatchedEquation>& result,
    Model<EquationType>& model,
    const mlir::BlockAndValueMapping& derivatives)
{
  Variables allVariables = model.getVariables();

  // Filter the variables. State and constant ones must not in fact
  // take part into the matching process as their values are already
  // determined (state variables depend on their derivatives, while
  // constants have a fixed value).

  Variables variables;

  for (const auto& variable : allVariables) {
    mlir::Value var = variable->getValue();

    if (!derivatives.contains(var)) {
      auto nonStateVariable = std::make_unique<Variable>(var);

      if (!nonStateVariable->isConstant()) {
        variables.add(std::move(nonStateVariable));
      }
    }
  }

  model.setVariables(variables);
  result.setVariables(variables);

  model.getEquations().setVariables(model.getVariables());

  MatchingGraph<Variable*, Equation*> matchingGraph;

  for (const auto& variable : model.getVariables()) {
    matchingGraph.addVariable(variable.get());
  }

  for (const auto& equation : model.getEquations()) {
    matchingGraph.addEquation(equation.get());
  }

  // Apply the simplification algorithm to solve the obliged matches
  if (!matchingGraph.simplify()) {
    model.getOperation().emitError("Inconsistency found during the matching simplification process");
    return mlir::failure();
  }

  // Apply the full matching algorithm for the equations and variables that are still unmatched
  if (!matchingGraph.match()) {
    model.getOperation().emitError("Matching failed");
    return mlir::failure();
  }

  Equations<MatchedEquation> matchedEquations;

  for (auto& solution : matchingGraph.getMatch()) {
    auto clone = solution.getEquation()->clone();
    auto matchedEquation = std::make_unique<MatchedEquation>(std::move(clone), solution.getAccess());

    for (size_t i = 0, e = solution.getEquation()->getNumOfIterationVars(); i < e; ++i) {
      matchedEquation->setMatchedIndexes(i, solution.getRangeBegin(i), solution.getRangeEnd(i));
    }

    matchedEquations.add(std::move(matchedEquation));
  }

  result.setEquations(matchedEquations);
  return mlir::success();
}

static mlir::LogicalResult replaceAccesses(
    mlir::OpBuilder& builder,
    Equation& destination,
    const AccessFunction& accessFunction,
    const EquationPath& accessPath,
    const MatchedEquation& source)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto writeAccess = source.getWrite();
  const auto& writeAccessFunction = writeAccess.getAccessFunction();

  if (!writeAccessFunction.isInvertible()) {
    source.getOperation().emitError("The write access is not invertible");
    return mlir::failure();
  }

  auto inverseWriteFunction = writeAccessFunction.inverse();
  auto transformation = accessFunction.combine(inverseWriteFunction);

  if (auto res = source.replaceInto(builder, destination, accessFunction, accessPath, source.getWrite()); mlir::failed(res)) {
    return res;
  }

  return mlir::success();
}

/// Modify the IR in order to solve the algebraic loops
static mlir::LogicalResult solveAlgebraicLoops(
    Model<MatchedEquation>& model, mlir::OpBuilder& builder)
{
  std::vector<MatchedEquation*> equations;

  for (const auto& equation : model.getEquations()) {
    equations.push_back(equation.get());
  }

  CyclesFinder<Variable*, MatchedEquation*> cyclesFinder(equations);
  auto cycles = cyclesFinder.getEquationsCycles();

  // The new equations without cycles
  Equations<MatchedEquation> newEquations;

  for (const auto& cycle : cycles) {
    for (const auto& interval : cycle) {
      auto clonedEquation = Equation::build(
          cycle.getEquation()->cloneIR(),
          cycle.getEquation()->getVariables());

      for (const auto& dependency : interval.getDestinations()) {
        const auto& access = dependency.getAccess();
        const auto& destination = dependency.getNode();

        if (auto res = replaceAccesses(
            builder, *clonedEquation,
            access.getAccessFunction(),
            access.getProperty(),
            *destination.getEquation()); mlir::failed(res)) {
          return res;
        }
      }

      // Create the matched equation on the cloned operation
      auto matchedEquation = std::make_unique<MatchedEquation>(
          std::move(clonedEquation), EquationPath(EquationPath::LEFT));

      for (size_t i = 0, e = cycle.getEquation()->getNumOfIterationVars(); i < e; ++i) {
        matchedEquation->setMatchedIndexes(
            i, cycle.getEquation()->getRangeBegin(i), cycle.getEquation()->getRangeEnd(i));
      }

      newEquations.add(std::move(matchedEquation));

      // TODO process multiple levels
    }
  }

  // Add the equations which had no cycle
  std::set<MatchedEquation*> equationsWithCycles;

  for (const auto& cycle : cycles) {
    equationsWithCycles.insert(cycle.getEquation());
  }

  // TODO: add equations with no cycles. And also add the indices without loops of the equations with cycles

  // Erase the original equations with cycles
  for (auto& equation : model.getEquations()) {
    if (equationsWithCycles.find(equation.get()) != equationsWithCycles.end()) {
      equation->eraseIR();
    }
  }

  // Set the new equations of the model
  model.setEquations(newEquations);

  return mlir::success();
}

/// Schedule the equations.
static mlir::LogicalResult scheduling(
    Model<ScheduledEquation>& result, const Model<MatchedEquation>& model)
{
  result.setVariables(model.getVariables());
  std::vector<MatchedEquation*> equations;

  for (const auto& equation : model.getEquations()) {
    equations.push_back(equation.get());
  }

  Scheduler<Variable*, MatchedEquation*> scheduler;
  Equations<ScheduledEquation> scheduledEquations;

  for (const auto& solution : scheduler.schedule(equations)) {
    auto clone = std::make_unique<MatchedEquation>(*solution.getEquation());
    auto scheduledEquation = std::make_unique<ScheduledEquation>(std::move(clone), solution.getIterationDirection());

    for (size_t i = 0, e = solution.getEquation()->getNumOfIterationVars(); i < e; ++i) {
      scheduledEquation->setScheduledIndexes(i, solution.getRangeBegin(i), solution.getRangeEnd(i));
    }

    scheduledEquations.push_back(std::move(scheduledEquation));
  }

  result.setEquations(std::move(scheduledEquations));
  return mlir::success();
}

/// Model solving pass.
/// Its objective is to convert a descriptive (and thus not sequential) model
/// into an algorithmic one and to create the functions controlling the simulation.
class SolveModelPass: public mlir::PassWrapper<SolveModelPass, mlir::OperationPass<ModelOp>>
{
	public:
    explicit SolveModelPass(SolveModelOptions options, unsigned int bitWidth)
        : options(std::move(options)),
          bitWidth(std::move(bitWidth))
    {
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override
    {
      registry.insert<ModelicaDialect>();
      registry.insert<mlir::scf::SCFDialect>();
      registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnOperation() override
    {
      // Group chained subscriptions into a single one
      if (mlir::failed(mergeChainedSubscriptions())) {
        return signalPassFailure();
      }

      Model model(getOperation());
      mlir::OpBuilder builder(model.getOperation());

      // Remove the derivative operations and allocate the appropriate memory buffers
      mlir::BlockAndValueMapping derivatives;

      if (mlir::failed(removeDerivatives(builder, model, derivatives))) {
        model.getOperation().emitError("Derivative could not be converted to variables");
        return signalPassFailure();
      }

      // Now that the additional variables have been created, we can start a discovery process
      model.setVariables(discoverVariables(model.getOperation()));
      model.setEquations(discoverEquations(model.getOperation(), model.getVariables()));

      // Matching process
      Model<MatchedEquation> matchedModel(model.getOperation());

      if (mlir::failed(::matching(matchedModel, model, derivatives))) {
        return signalPassFailure();
      }

      llvm::errs() << "BEFORE LOOPS SOLVING\n";
      matchedModel.getOperation().dump();

      // Resolve the algebraic loops
      if (mlir::failed(::solveAlgebraicLoops(matchedModel, builder))) {
        return signalPassFailure();
      }

      llvm::errs() << "AFTER LOOPS SOLVING\n";
      matchedModel.getOperation().dump();

      // Schedule the equations
      Model<ScheduledEquation> scheduledModel(matchedModel.getOperation());

      if (mlir::failed(::scheduling(scheduledModel, matchedModel))) {
        return signalPassFailure();
      }

      // Create the simulation functions
      mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
      TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
      ModelConverter modelConverter(options, typeConverter);

      if (auto status = modelConverter.convert(builder, scheduledModel, derivatives); mlir::failed(status)) {
        return signalPassFailure();
      }
    }

  private:
    mlir::LogicalResult mergeChainedSubscriptions()
    {
      // TODO

      return mlir::success();
    }

	private:
	  SolveModelOptions options;
	  unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> marco::codegen::createSolveModelPass(SolveModelOptions options, unsigned int bitWidth)
{
	return std::make_unique<SolveModelPass>(options, bitWidth);
}
