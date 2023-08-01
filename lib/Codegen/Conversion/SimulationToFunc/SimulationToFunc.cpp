#include "marco/Codegen/Conversion/SimulationToFunc/SimulationToFunc.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "marco/Codegen/Runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "llvm/ADT/BitVector.h"

namespace mlir
{
#define GEN_PASS_DEF_SIMULATIONTOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::marco::codegen;
using namespace ::mlir::simulation;

/// Get the LLVM function with the given name, or declare it inside the module
/// if not present.
static mlir::LLVM::LLVMFuncOp declareExternalFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    llvm::StringRef name,
    mlir::LLVM::LLVMFunctionType type)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

namespace
{
  /// Generic rewrite pattern that provides some utility functions.
  template<typename Op>
  class SimulationOpRewritePattern : public mlir::OpRewritePattern<Op>
  {
    public:
      using mlir::OpRewritePattern<Op>::OpRewritePattern;

    protected:
      static constexpr size_t solversDataPosition = 0;
      static constexpr size_t timeVariablePosition = 1;
      static constexpr size_t variablesOffset = 2;

    protected:
      mlir::Type getVoidPtrType() const
      {
        mlir::Type i8Type = mlir::IntegerType::get(this->getContext(), 8);
        return mlir::LLVM::LLVMPointerType::get(i8Type);
      }

      /// Get the type of the runtime data structure.
      mlir::LLVM::LLVMStructType getRuntimeDataStructType(
          mlir::OpBuilder& builder, mlir::TypeRange variablesTypes) const
      {
        llvm::SmallVector<mlir::Type> structTypes;

        // Solvers-reserved data.
        structTypes.push_back(getVoidPtrType());

        // Time.
        structTypes.push_back(builder.getF64Type());

        // Variables.
        for (mlir::Type varType : variablesTypes) {
          structTypes.push_back(varType);
        }

        return mlir::LLVM::LLVMStructType::getLiteral(
            builder.getContext(), structTypes);
      }

      /// Load the data structure from the opaque pointer that is passed around
      /// the simulation functions.
      mlir::Value loadRuntimeDataFromOpaquePtr(
          mlir::OpBuilder& builder,
          mlir::Value opaquePtr,
          mlir::TypeRange variablesTypes) const
      {
        mlir::Location loc = opaquePtr.getLoc();

        mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(
            getRuntimeDataStructType(builder, variablesTypes));

        mlir::Value structPtr = builder.create<mlir::LLVM::BitcastOp>(
            loc, structPtrType, opaquePtr);

        return builder.create<mlir::LLVM::LoadOp>(loc, structPtr);
      }

      /// Store the data structure within the memory addressed given by the
      /// opaque pointer that is passed around the simulation functions.
      void setRuntimeDataIntoOpaquePtr(
          mlir::OpBuilder& builder,
          mlir::Value opaquePtr,
          mlir::Value data,
          mlir::TypeRange variablesTypes) const
      {
        mlir::Location loc = data.getLoc();

        auto runtimeDataStructPtrType = mlir::LLVM::LLVMPointerType::get(
            getRuntimeDataStructType(builder, variablesTypes));

        assert(data.getType() == runtimeDataStructPtrType.getElementType());

        mlir::Value runtimeDataStructPtr =
            builder.create<mlir::LLVM::BitcastOp>(
                loc, runtimeDataStructPtrType, opaquePtr);

        builder.create<mlir::LLVM::StoreOp>(loc, data, runtimeDataStructPtr);
      }

      /// Extract a value from a struct.
      mlir::Value extractValue(
          mlir::OpBuilder& builder,
          mlir::Value structValue,
          unsigned int position) const
      {
        mlir::Location loc = structValue.getLoc();

        assert(structValue.getType().isa<mlir::LLVM::LLVMStructType>() &&
            "Not an LLVM struct");

        auto structType =
            structValue.getType().cast<mlir::LLVM::LLVMStructType>();

        auto structTypes = structType.getBody();

        assert (position < structTypes.size() &&
               "LLVM struct: index is out of bounds");

        return builder.create<mlir::LLVM::ExtractValueOp>(
            loc, structTypes[position], structValue, position);
      }

      /// Get the type of the solvers structure.
      mlir::LLVM::LLVMStructType getSolversStructType(
          mlir::OpBuilder& builder,
          mlir::TypeRange solverTypes) const
      {
        llvm::SmallVector<mlir::Type, 1> structTypes(
            solverTypes.begin(), solverTypes.end());

        return mlir::LLVM::LLVMStructType::getLiteral(
            builder.getContext(), structTypes);
      }

      /// Load the solvers structure from the opaque pointer that lives within
      /// the runtime data structure.
      mlir::Value loadSolversFromRuntimeData(
          mlir::OpBuilder& builder,
          mlir::Value runtimeData,
          mlir::TypeRange solverTypes) const
      {
        mlir::Location loc = runtimeData.getLoc();

        mlir::Type structPtrType = mlir::LLVM::LLVMPointerType::get(
            getSolversStructType(builder, solverTypes));

        mlir::Value opaquePtr = extractValue(
            builder, runtimeData, solversDataPosition);

        mlir::Value structPtr = builder.create<mlir::LLVM::BitcastOp>(
            loc, structPtrType, opaquePtr);

        return builder.create<mlir::LLVM::LoadOp>(loc, structPtr);
      }

      mlir::Value extractSolver(
          mlir::OpBuilder& builder,
          mlir::Value solversData,
          unsigned int solverIndex) const
      {
        return extractValue(builder, solversData, solverIndex);
      }

      /// Extract the time variable from the data structure shared between the
      /// various simulation functions.
      mlir::Value extractTime(
          mlir::OpBuilder& builder, mlir::Value runtimeData) const
      {
        return extractValue(builder, runtimeData, timeVariablePosition);
      }

      /// Extract a variable from the data structure shared between the various
      /// simulation functions.
      mlir::Value extractVariable(
          mlir::OpBuilder& builder,
          mlir::Value runtimeData,
          unsigned int varIndex) const
      {
        return extractValue(builder, runtimeData, varIndex + variablesOffset);
      }

      /// Create the instructions to allocate some data with a given type.
      mlir::Value alloc(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Location loc,
          mlir::Type type) const
      {
        // Add the function to the module.
        mlir::LLVM::LLVMFuncOp allocFn = lookupOrCreateHeapAllocFn(
            module, builder.getI64Type());

        // Determine the size (in bytes) of the memory to be allocated.
        mlir::Type ptrType = mlir::LLVM::LLVMPointerType::get(type);
        mlir::Value nullPtr = builder.create<mlir::LLVM::NullOp>(loc, ptrType);

        mlir::Value one = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getI64IntegerAttr(1));

        mlir::Value gepPtr = builder.create<mlir::LLVM::GEPOp>(
            loc, ptrType, nullPtr, one);

        mlir::Value sizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(
            loc, builder.getI64Type(), gepPtr);

        auto callOp =
            builder.create<mlir::LLVM::CallOp>(loc, allocFn, sizeBytes);

        mlir::Value resultOpaquePtr = callOp.getResult();

        // Cast the allocated memory pointer to a pointer of the original type.
        return builder.create<mlir::LLVM::BitcastOp>(
            loc, ptrType, resultOpaquePtr);
      }

      mlir::Value createGlobalString(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          mlir::StringRef name,
          mlir::StringRef value) const
      {
        mlir::LLVM::GlobalOp global;

        {
          // Create the global at the entry of the module.
          mlir::OpBuilder::InsertionGuard insertGuard(builder);
          builder.setInsertionPointToStart(moduleOp.getBody());

          auto type = mlir::LLVM::LLVMArrayType::get(
              mlir::IntegerType::get(
                  builder.getContext(), 8), value.size() + 1);

          global = builder.create<mlir::LLVM::GlobalOp>(
              loc, type, true, mlir::LLVM::Linkage::Internal, name,
              builder.getStringAttr(llvm::StringRef(
                  value.data(), value.size() + 1)));
        }

        // Get the pointer to the first character of the global string.
        mlir::Value globalPtr =
            builder.create<mlir::LLVM::AddressOfOp>(loc, global);

        mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
            loc,
            mlir::IntegerType::get(builder.getContext(), 64),
            builder.getIntegerAttr(builder.getIndexType(), 0));

        return builder.create<mlir::LLVM::GEPOp>(
            loc,
            getVoidPtrType(),
            globalPtr, llvm::makeArrayRef({cst0, cst0}));
      }
  };
}

namespace
{
  struct InitFunctionOpLowering
      : public SimulationOpRewritePattern<InitFunctionOp>
  {
    using SimulationOpRewritePattern<InitFunctionOp>
        ::SimulationOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        InitFunctionOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      rewriter.setInsertionPointToEnd(moduleOp.getBody());

      auto funcOp = rewriter.create<mlir::func::FuncOp>(
          op.getLoc(), "init",
          rewriter.getFunctionType(llvm::None, getVoidPtrType()));

      mlir::BlockAndValueMapping mapping;

      rewriter.cloneRegionBefore(
          op.getBodyRegion(),
          funcOp.getFunctionBody(),
          funcOp.getFunctionBody().end(),
          mapping);

      auto terminator = mlir::cast<YieldOp>(
          funcOp.getFunctionBody().back().getTerminator());

      rewriter.setInsertionPoint(terminator);

      // Create the runtime data structure.
      for (YieldOp yieldOp :
           llvm::make_early_inc_range(
               funcOp.getFunctionBody().getOps<YieldOp>())) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(yieldOp);

        auto runtimeDataStructType = getRuntimeDataStructType(
            rewriter, yieldOp.getValues().getTypes());

        mlir::Value runtimeDataStruct =
            rewriter.create<mlir::LLVM::UndefOp>(
                yieldOp.getLoc(), runtimeDataStructType);

        for (const auto& var : llvm::enumerate(yieldOp.getValues())) {
          runtimeDataStruct = rewriter.create<mlir::LLVM::InsertValueOp>(
              yieldOp.getLoc(),
              runtimeDataStruct,
              var.value(),
              var.index() + variablesOffset);
        }

        // Allocate the main runtime data structure.
        mlir::Value runtimeDataStructPtr = alloc(
            rewriter, moduleOp, yieldOp.getLoc(), runtimeDataStructType);

        rewriter.create<mlir::LLVM::StoreOp>(
            yieldOp.getLoc(),
            runtimeDataStruct, runtimeDataStructPtr);

        mlir::Value runtimeDataOpaquePtr =
            rewriter.create<mlir::LLVM::BitcastOp>(
                yieldOp.getLoc(), getVoidPtrType(), runtimeDataStructPtr);

        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
            yieldOp, runtimeDataOpaquePtr);
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  struct DeinitFunctionOpLowering
      : public SimulationOpRewritePattern<DeinitFunctionOp>
  {
    using SimulationOpRewritePattern<DeinitFunctionOp>
        ::SimulationOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        DeinitFunctionOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      rewriter.setInsertionPointToEnd(moduleOp.getBody());

      auto funcOp = rewriter.create<mlir::func::FuncOp>(
          op.getLoc(), "deinit",
          rewriter.getFunctionType(getVoidPtrType(), llvm::None));

      mlir::Block* entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::Value opaquePtr = funcOp.getArgument(0);

      mlir::Value runtimeData = loadRuntimeDataFromOpaquePtr(
          rewriter, opaquePtr, op.getVariables().getTypes());

      mlir::BlockAndValueMapping mapping;

      // Extract the variables.
      for (const auto& var : llvm::enumerate(op.getVariables())) {
        mlir::Value mapped = extractVariable(
            rewriter, runtimeData, var.index());

        mapping.map(var.value(), mapped);
      }

      // Clone the blocks.
      rewriter.cloneRegionBefore(
          op.getBodyRegion(),
          funcOp.getFunctionBody(),
          funcOp.getFunctionBody().end(),
          mapping);

      // Create the branch from the first block (used for casts) to the
      // original first block.
      if (!op.getBodyRegion().empty()) {
        rewriter.setInsertionPointToEnd(entryBlock);

        rewriter.create<mlir::cf::BranchOp>(
            loc, mapping.lookup(&op.getBodyRegion().front()));
      }

      // Terminate the body of the function.
      for (YieldOp yieldOp :
           llvm::make_early_inc_range(
               funcOp.getFunctionBody().getOps<YieldOp>())) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(yieldOp);

        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
            yieldOp, yieldOp.getValues());
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  template<typename Op>
  struct InitSolversFunctionOpLowering : public SimulationOpRewritePattern<Op>
  {
    using SimulationOpRewritePattern<Op>::SimulationOpRewritePattern;

    virtual llvm::StringRef getFunctionName() const = 0;

    mlir::LogicalResult matchAndRewrite(
        Op op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      // Get the parent simulation module.
      auto simulationModuleOp =
          op->template getParentOfType<mlir::simulation::ModuleOp>();

      auto variableTypes = simulationModuleOp.getVariablesTypes();

      // Create the function at the end of the module.
      auto moduleOp = op->template getParentOfType<mlir::ModuleOp>();
      rewriter.setInsertionPointToEnd(moduleOp.getBody());

      auto funcOp = rewriter.create<mlir::func::FuncOp>(
          op.getLoc(), getFunctionName(),
          rewriter.getFunctionType(this->getVoidPtrType(), llvm::None));

      // Create the entry block.
      mlir::Block* entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::Value runtimeDataOpaquePtr = funcOp.getArgument(0);

      mlir::Value runtimeData = this->loadRuntimeDataFromOpaquePtr(
          rewriter, runtimeDataOpaquePtr, variableTypes);

      mlir::BlockAndValueMapping mapping;

      // Extract the variables.
      for (const auto& variable : llvm::enumerate(op.getVariables())) {
        mlir::Value mapped = this->extractVariable(
            rewriter, runtimeData, variable.index());

        mapping.map(variable.value(), mapped);
      }

      rewriter.cloneRegionBefore(
          op.getBodyRegion(),
          funcOp.getFunctionBody(),
          funcOp.getFunctionBody().end(),
          mapping);

      // Create the branch from the first block (used for casts) to the
      // original first block.
      if (!op.getBodyRegion().empty()) {
        rewriter.setInsertionPointToEnd(entryBlock);

        rewriter.create<mlir::cf::BranchOp>(
            loc, mapping.lookup(&op.getBodyRegion().front()));
      }

      // Create the solvers structure.
      for (YieldOp yieldOp :
           llvm::make_early_inc_range(
               funcOp.getFunctionBody().template getOps<YieldOp>())) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(yieldOp);

        auto solversStructType = this->getSolversStructType(
            rewriter, yieldOp.getValues().getTypes());

        mlir::Value solversStruct =
            rewriter.create<mlir::LLVM::UndefOp>(
                yieldOp.getLoc(), solversStructType);

        for (const auto& var : llvm::enumerate(yieldOp.getValues())) {
          solversStruct = rewriter.create<mlir::LLVM::InsertValueOp>(
              yieldOp.getLoc(), solversStruct, var.value(), var.index());
        }

        // Allocate the solvers data structure.
        mlir::Value solversStructPtr = this->alloc(
            rewriter, moduleOp, yieldOp.getLoc(), solversStructType);

        rewriter.create<mlir::LLVM::StoreOp>(
            yieldOp.getLoc(), solversStruct, solversStructPtr);

        // Cast it to an opaque pointer and store it into the runtime data
        // structure.
        mlir::Value solversStructOpaquePtr =
            rewriter.create<mlir::LLVM::BitcastOp>(
                yieldOp.getLoc(), this->getVoidPtrType(), solversStructPtr);

        runtimeData = rewriter.create<mlir::LLVM::InsertValueOp>(
            yieldOp.getLoc(),
            runtimeData,
            solversStructOpaquePtr,
            this->solversDataPosition);

        this->setRuntimeDataIntoOpaquePtr(
            rewriter, runtimeDataOpaquePtr, runtimeData, variableTypes);

        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(yieldOp);
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  struct InitICSolversFunctionOpLowering
      : public InitSolversFunctionOpLowering<InitICSolversFunctionOp>
  {
    using InitSolversFunctionOpLowering<InitICSolversFunctionOp>
        ::InitSolversFunctionOpLowering;

    llvm::StringRef getFunctionName() const override
    {
      return "initICSolvers";
    }
  };

  struct InitMainSolversFunctionOpLowering
      : public InitSolversFunctionOpLowering<InitMainSolversFunctionOp>
  {
    using InitSolversFunctionOpLowering<InitMainSolversFunctionOp>
        ::InitSolversFunctionOpLowering;

    llvm::StringRef getFunctionName() const override
    {
      return "initMainSolvers";
    }
  };

  template<typename Op>
  struct DeinitSolversFunctionOpLowering : public SimulationOpRewritePattern<Op>
  {
    using SimulationOpRewritePattern<Op>::SimulationOpRewritePattern;

    virtual llvm::StringRef getFunctionName() const = 0;

    mlir::LogicalResult matchAndRewrite(
        Op op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      auto moduleOp = op->template getParentOfType<mlir::ModuleOp>();
      rewriter.setInsertionPointToEnd(moduleOp.getBody());

      auto funcOp = rewriter.create<mlir::func::FuncOp>(
          op.getLoc(), getFunctionName(),
          rewriter.getFunctionType(this->getVoidPtrType(), llvm::None));

      mlir::Block* entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::Value runtimeDataOpaquePtr = funcOp.getArgument(0);

      auto simulationModuleOp =
          op->template getParentOfType<mlir::simulation::ModuleOp>();

      auto variableTypes = simulationModuleOp.getVariablesTypes();

      mlir::Value runtimeData = this->loadRuntimeDataFromOpaquePtr(
          rewriter, runtimeDataOpaquePtr, variableTypes);

      mlir::BlockAndValueMapping mapping;

      // Extract the solvers.
      mlir::Value solversData = this->loadSolversFromRuntimeData(
          rewriter, runtimeData, op.getSolverTypes());

      for (const auto& solver : llvm::enumerate(op.getSolvers())) {
        mlir::Value mapped = this->extractSolver(
            rewriter, solversData, solver.index());

        mapping.map(solver.value(), mapped);
      }

      // Clone the blocks.
      rewriter.cloneRegionBefore(
          op.getBodyRegion(),
          funcOp.getFunctionBody(),
          funcOp.getFunctionBody().end(),
          mapping);

      // Create the branch from the first block (used for casts) to the
      // original first block.
      if (!op.getBodyRegion().empty()) {
        rewriter.setInsertionPointToEnd(entryBlock);

        rewriter.create<mlir::cf::BranchOp>(
            loc, mapping.lookup(&op.getBodyRegion().front()));
      }

      // Terminate the body of the function.
      for (YieldOp yieldOp :
           llvm::make_early_inc_range(
               funcOp.getFunctionBody().template getOps<YieldOp>())) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(yieldOp);

        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
            yieldOp, yieldOp.getValues());
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  struct DeinitICSolversFunctionOpLowering
      : public DeinitSolversFunctionOpLowering<DeinitICSolversFunctionOp>
  {
    using DeinitSolversFunctionOpLowering<DeinitICSolversFunctionOp>
        ::DeinitSolversFunctionOpLowering;

    llvm::StringRef getFunctionName() const override
    {
      return "deinitICSolvers";
    }
  };

  struct DeinitMainSolversFunctionOpLowering
      : public DeinitSolversFunctionOpLowering<DeinitMainSolversFunctionOp>
  {
    using DeinitSolversFunctionOpLowering<DeinitMainSolversFunctionOp>
        ::DeinitSolversFunctionOpLowering;

    llvm::StringRef getFunctionName() const override
    {
      return "deinitMainSolvers";
    }
  };

  struct FunctionOpLowering : public SimulationOpRewritePattern<FunctionOp>
  {
    using SimulationOpRewritePattern<FunctionOp>::SimulationOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        FunctionOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      rewriter.setInsertionPointToEnd(moduleOp.getBody());

      llvm::SmallVector<mlir::Type, 1> argTypes;
      argTypes.push_back(getVoidPtrType());

      for (mlir::Type extraArgType : op.getExtraArgTypes()) {
        argTypes.push_back(extraArgType);
      }

      auto funcOp = rewriter.create<mlir::func::FuncOp>(
          op.getLoc(), op.getSymName(),
          rewriter.getFunctionType(argTypes, op.getResultTypes()));

      mlir::Block* entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      mlir::Value runtimeDataOpaquePtr = funcOp.getArgument(0);

      auto simulationModuleOp =
          op->getParentOfType<mlir::simulation::ModuleOp>();

      auto variableTypes = simulationModuleOp.getVariablesTypes();

      mlir::Value runtimeData = loadRuntimeDataFromOpaquePtr(
          rewriter, runtimeDataOpaquePtr, variableTypes);

      mlir::BlockAndValueMapping mapping;

      // Extract the solvers.
      mlir::Value solversData = loadSolversFromRuntimeData(
          rewriter, runtimeData, op.getSolverTypes());

      for (const auto& solver : llvm::enumerate(op.getSolvers())) {
        mlir::Value mapped = extractSolver(
            rewriter, solversData, solver.index());

        mapping.map(solver.value(), mapped);
      }

      // Extract the time variable.
      mlir::Value time = extractTime(rewriter, runtimeData);
      mapping.map(op.getTime(), time);

      // Extract the variables.
      for (const auto& variable : llvm::enumerate(op.getVariables())) {
        mlir::Value mapped = extractVariable(
            rewriter, runtimeData, variable.index());

        mapping.map(variable.value(), mapped);
      }

      // Extract the extra arguments.
      for (const auto& arg : llvm::enumerate(op.getExtraArgs())) {
        mlir::Value mapped = funcOp.getArgument(1 + arg.index());
        mapping.map(arg.value(), mapped);
      }

      // Clone the blocks.
      rewriter.cloneRegionBefore(
          op.getBodyRegion(),
          funcOp.getFunctionBody(),
          funcOp.getFunctionBody().end(),
          mapping);

      // Create the branch from the first block (used for casts) to the
      // original first block.
      if (!op.getBodyRegion().empty()) {
        rewriter.setInsertionPointToEnd(entryBlock);

        rewriter.create<mlir::cf::BranchOp>(
            loc, mapping.lookup(&op.getBodyRegion().front()));
      }

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };

  struct ReturnOpLowering : public SimulationOpRewritePattern<ReturnOp>
  {
    using SimulationOpRewritePattern<ReturnOp>::SimulationOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        ReturnOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, op.getOperands());
      return mlir::success();
    }
  };
}

static void populateSimulationToFuncFunctionLikeConversionPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::MLIRContext* context)
{
  patterns.insert<
      InitFunctionOpLowering,
      DeinitFunctionOpLowering,
      InitICSolversFunctionOpLowering,
      InitMainSolversFunctionOpLowering,
      DeinitICSolversFunctionOpLowering,
      DeinitMainSolversFunctionOpLowering,
      FunctionOpLowering,
      ReturnOpLowering>(context);
}

namespace
{
  class ModuleOpLowering
      : public SimulationOpRewritePattern<mlir::simulation::ModuleOp>
  {
    public:
      using SimulationOpRewritePattern<mlir::simulation::ModuleOp>
          ::SimulationOpRewritePattern;

      ModuleOpLowering(mlir::MLIRContext* context, bool emitMainFunction)
          : SimulationOpRewritePattern(context),
            emitMainFunction(emitMainFunction)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          mlir::simulation::ModuleOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        createGetModelNameFunction(rewriter, op);
        createGetNumOfVariablesFunction(rewriter, op);
        createGetVariableNamesFunction(rewriter, op);
        createGetVariableRankFunction(rewriter, op);
        createIsPrintableFunction(rewriter, op);
        createGetVariableNumOfPrintableRangesFunction(rewriter, op);
        createGetVariablePrintableRangeBeginFunction(rewriter, op);
        createGetVariablePrintableRangeEndFunction(rewriter, op);
        createGetDerivativeFunction(rewriter, op);
        createGetTimeFunction(rewriter, op);
        createSetTimeFunction(rewriter, op);
        createGetVariableValueFunction(rewriter, op);

        if (emitMainFunction) {
          createMainFunction(rewriter, op);
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }

    private:
      /// Create the function to be called to retrieve the name of the compiled
      /// model.
      void createGetModelNameFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to retrieve the number of variables
      /// of the compiled model.
      void createGetNumOfVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to retrieve the name of variables of
      /// the compiled model.
      void createGetVariableNamesFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to retrieve the name of variables of
      /// the compiled model.
      void createGetVariableRankFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to determine if a variable has at
      /// least one scalar variable to be printed.
      void createIsPrintableFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to retrieve the number of printable
      /// indices ranges for a given variable.
      void createGetVariableNumOfPrintableRangesFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to retrieve the begin index of a
      /// printable range for a given variable and dimension.
      void createGetVariablePrintableRangeBeginFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to retrieve the end index of a
      /// printable range for a given variable and dimension.
      void createGetVariablePrintableRangeEndFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      void createGetVariablePrintableRangeBoundariesFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          llvm::StringRef functionName,
          std::function<int64_t(const std::pair<int64_t, int64_t>&)>
              boundaryGetterCallback) const;

      /// Create the function to be called to retrieve the index of the
      /// derivative of a variable.
      void createGetDerivativeFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to retrieve the current time of the
      /// simulation.
      void createGetTimeFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to set the current time of the
      /// simulation.
      void createSetTimeFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      /// Create the function to be called to retrieve the value of a scalar
      /// variable.
      void createGetVariableValueFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

      mlir::func::FuncOp convertVariableGetterOp(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp,
          VariableGetterOp op,
          size_t& convertedGetters) const;

      /// Create the main function, which is called when the executable of the
      /// simulation is run. In order to keep the code generation simpler, the
      /// real implementation of the function managing the simulation lives
      /// within the runtime library and the main just consists in a call to
      /// such function.
      void createMainFunction(
          mlir::OpBuilder& builder,
          mlir::simulation::ModuleOp simulationModuleOp) const;

    private:
      bool emitMainFunction;
  };
}

void ModuleOpLowering::createGetModelNameFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto function = builder.create<mlir::func::FuncOp>(
      loc, "getModelName",
      builder.getFunctionType(llvm::None, getVoidPtrType()));

  auto* entryBlock = function.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Create a global string containing the name of the model.
  mlir::Value result = createGlobalString(
      builder, loc, moduleOp, "modelName",
      simulationModuleOp.getModelName());

  builder.create<mlir::func::ReturnOp>(loc, result);
}

void ModuleOpLowering::createGetNumOfVariablesFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto module = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(module.getBody());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "getNumOfVariables",
      builder.getFunctionType(llvm::None, builder.getI64Type()));

  mlir::Block* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Create the result.
  size_t numOfVariables = simulationModuleOp.getVariables().size();

  mlir::Value result = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(numOfVariables));

  builder.create<mlir::func::ReturnOp>(loc, result);
}

void ModuleOpLowering::createGetVariableNamesFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // char* type.
  mlir::Type charPtrType = mlir::LLVM::LLVMPointerType::get(
      mlir::IntegerType::get(builder.getContext(), 8));

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "getVariableName",
      builder.getFunctionType(builder.getI64Type(), charPtrType));

  // Create the entry block.
  mlir::Block* entryBlock = funcOp.addEntryBlock();

  // Create the last block receiving the value to be returned.
  mlir::Block* returnBlock = builder.createBlock(
      &funcOp.getFunctionBody(),
      funcOp.getFunctionBody().end(),
      charPtrType,
      loc);

  builder.setInsertionPointToEnd(returnBlock);
  builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

  // Create the blocks and the switch.
  auto variableAttrs = simulationModuleOp.getVariables();

  size_t numCases = variableAttrs.size();
  llvm::SmallVector<int64_t> caseValues(numCases);
  llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
  llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

  for (size_t i = 0; i < numCases; ++i) {
    caseValues[i] = i;
    caseBlocks[i] = builder.createBlock(returnBlock);
    caseOperandsRefs[i] = llvm::None;
  }

  builder.setInsertionPointToStart(entryBlock);

  mlir::Value unknownVariableName = createGlobalString(
      builder, loc, moduleOp, "varUnknown", "");

  builder.create<mlir::cf::SwitchOp>(
      loc,
      entryBlock->getArgument(0), returnBlock, unknownVariableName,
      builder.getI64TensorAttr(caseValues),
      caseBlocks, caseOperandsRefs);

  // Populate the case blocks.
  for (const auto& variable :
       llvm::enumerate(variableAttrs.getAsRange<VariableAttr>())) {
    size_t i = variable.index();
    builder.setInsertionPointToStart(caseBlocks[i]);

    std::string symbolName = "var" + std::to_string(variable.index());
    mlir::Value variableName = unknownVariableName;

    if (llvm::StringRef nameStr = variable.value().getName();
        !nameStr.empty()) {
      variableName = createGlobalString(
          builder, loc, moduleOp, symbolName, nameStr);
    }

    builder.create<mlir::cf::BranchOp>(loc, returnBlock, variableName);
  }
}

void ModuleOpLowering::createGetVariableRankFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "getVariableRank",
      builder.getFunctionType(builder.getI64Type(), builder.getI64Type()));

  // Create the entry block.
  mlir::Block* entryBlock = funcOp.addEntryBlock();

  // Create the last block receiving the value to be returned.
  mlir::Block* returnBlock = builder.createBlock(
      &funcOp.getFunctionBody(),
      funcOp.getFunctionBody().end(),
      builder.getI64Type(), loc);

  builder.setInsertionPointToEnd(returnBlock);
  builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

  // Create the blocks and the switch.
  auto variableAttrs = simulationModuleOp.getVariables();

  size_t numCases = variableAttrs.size();
  llvm::SmallVector<int64_t> caseValues(numCases);
  llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
  llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

  for (size_t i = 0; i < numCases; ++i) {
    caseValues[i] = i;
    caseBlocks[i] = builder.createBlock(returnBlock);
    caseOperandsRefs[i] = llvm::None;
  }

  builder.setInsertionPointToStart(entryBlock);

  mlir::Value defaultOperand = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(0));

  builder.create<mlir::cf::SwitchOp>(
      loc,
      entryBlock->getArgument(0), returnBlock, defaultOperand,
      builder.getI64TensorAttr(caseValues),
      caseBlocks, caseOperandsRefs);

  builder.setInsertionPointToStart(entryBlock);

  // Populate the case blocks.
  for (const auto& variable :
       llvm::enumerate(variableAttrs.getAsRange<VariableAttr>())) {
    size_t i = variable.index();
    builder.setInsertionPointToStart(caseBlocks[i]);

    size_t rank = variable.value().getDimensions().size();

    mlir::Value result = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(rank));

    builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
  }
}

void ModuleOpLowering::createIsPrintableFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "isPrintable",
      builder.getFunctionType(builder.getI64Type(), builder.getI1Type()));

  // Create the entry block.
  mlir::Block* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value falseValue = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getBoolAttr(false));

  // Create the last block receiving the value to be returned.
  mlir::Block* returnBlock = builder.createBlock(
      &funcOp.getFunctionBody(),
      funcOp.getFunctionBody().end(),
      builder.getI1Type(), loc);

  builder.setInsertionPointToEnd(returnBlock);
  builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

  // Create the blocks and the switch.
  mlir::ArrayAttr variables = simulationModuleOp.getVariables();
  llvm::DenseSet<int64_t> printableVariables;

  for (const auto& variable :
       llvm::enumerate(variables.getAsRange<VariableAttr>())) {
    if (variable.value().getPrintable()) {
      printableVariables.insert(variable.index());
    }
  }

  llvm::SmallVector<int64_t> caseValues;
  llvm::SmallVector<mlir::Block*> caseBlocks;
  llvm::SmallVector<mlir::ValueRange> caseOperandsRefs;

  if (!printableVariables.empty()) {
    mlir::Block* printableVariableBlock = builder.createBlock(returnBlock);
    builder.setInsertionPointToStart(printableVariableBlock);

    mlir::Value trueValue = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getBoolAttr(true));

    builder.create<mlir::cf::BranchOp>(loc, returnBlock, trueValue);

    for (int64_t variablePos : printableVariables) {
      caseValues.push_back(variablePos);
      caseBlocks.push_back(printableVariableBlock);
      caseOperandsRefs.push_back(llvm::None);
    }
  }

  builder.setInsertionPointToEnd(entryBlock);

  builder.create<mlir::cf::SwitchOp>(
      loc,
      entryBlock->getArgument(0), returnBlock, falseValue,
      builder.getI64TensorAttr(caseValues),
      caseBlocks, caseOperandsRefs);
}

void ModuleOpLowering::createGetVariableNumOfPrintableRangesFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "getVariableNumOfPrintableRanges",
      builder.getFunctionType(builder.getI64Type(), builder.getI64Type()));

  // Create the entry block.
  mlir::Block* entryBlock = funcOp.addEntryBlock();

  // Create the last block receiving the value to be returned.
  mlir::Block* returnBlock = builder.createBlock(
      &funcOp.getFunctionBody(),
      funcOp.getFunctionBody().end(),
      builder.getI64Type(),
      loc);

  builder.setInsertionPointToEnd(returnBlock);
  builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

  // Collect the results.
  llvm::DenseMap<int64_t, llvm::DenseSet<int64_t>> resultsVariablesMap;

  for (const auto& variable : llvm::enumerate(
           simulationModuleOp.getVariables().getAsRange<VariableAttr>())) {
    if (variable.value().getPrintable() &&
        variable.value().getDimensions().size() != 0) {
      size_t rangesAmount = variable.value().getPrintableIndices().size();
      resultsVariablesMap[rangesAmount].insert(variable.index());
    }
  }

  // Create the blocks and the switch.
  llvm::SmallVector<int64_t> caseValues;
  llvm::SmallVector<mlir::Block*> caseBlocks;
  llvm::SmallVector<mlir::ValueRange> caseOperandsRefs;

  for (const auto& entry : resultsVariablesMap) {
    int64_t numOfRanges = entry.getFirst();

    mlir::Block* caseBlock = builder.createBlock(returnBlock);
    builder.setInsertionPointToStart(caseBlock);

    mlir::Value result = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(numOfRanges));

    builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);

    for (int64_t variable : entry.getSecond()) {
      caseValues.push_back(variable);
      caseBlocks.push_back(caseBlock);
      caseOperandsRefs.push_back(llvm::None);
    }
  }

  builder.setInsertionPointToStart(entryBlock);

  mlir::Value defaultOperand = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(0));

  builder.create<mlir::cf::SwitchOp>(
      loc,
      entryBlock->getArgument(0), returnBlock, defaultOperand,
      builder.getI64TensorAttr(caseValues),
      caseBlocks, caseOperandsRefs);
}

void ModuleOpLowering::createGetVariablePrintableRangeBeginFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  auto callback = [](const std::pair<int64_t, int64_t>& range) -> int64_t {
    return range.first;
  };

  return createGetVariablePrintableRangeBoundariesFunction(
      builder, simulationModuleOp,
      "getVariablePrintableRangeBegin",
      callback);
}

void ModuleOpLowering::createGetVariablePrintableRangeEndFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  auto callback = [](const std::pair<int64_t, int64_t>& range) -> int64_t {
    return range.second;
  };

  return createGetVariablePrintableRangeBoundariesFunction(
      builder, simulationModuleOp,
      "getVariablePrintableRangeEnd",
      callback);
}

void ModuleOpLowering::createGetVariablePrintableRangeBoundariesFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    llvm::StringRef functionName,
    std::function<int64_t(const std::pair<int64_t, int64_t>&)>
        boundaryGetterCallback) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  llvm::SmallVector<mlir::Type, 3> argTypes;
  argTypes.push_back(builder.getI64Type());
  argTypes.push_back(builder.getI64Type());
  argTypes.push_back(builder.getI64Type());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, functionName,
      builder.getFunctionType(argTypes, builder.getI64Type()));

  // Create the entry block.
  mlir::Block* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value unknownResult = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(-1));

  // Create the last block receiving the value to be returned.
  mlir::Block* returnBlock = builder.createBlock(
      &funcOp.getFunctionBody(),
      funcOp.getFunctionBody().end(),
      builder.getI64Type(),
      loc);

  builder.setInsertionPointToEnd(returnBlock);
  builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

  // Create the blocks and the switches.
  llvm::DenseSet<std::pair<int64_t, int64_t>> uniqueRanges;
  llvm::DenseSet<MultidimensionalRangeAttr> uniqueMultidimensionalRanges;

  for (const auto& variable : llvm::enumerate(
           simulationModuleOp.getVariables().getAsRange<VariableAttr>())) {
    for (MultidimensionalRangeAttr multidimensionalRange :
         variable.value().getPrintableIndices()) {
      uniqueMultidimensionalRanges.insert(multidimensionalRange);

      for (const auto& range : multidimensionalRange.getRanges()) {
        uniqueRanges.insert(range);
      }
    }
  }

  llvm::DenseMap<std::pair<int64_t, int64_t>, mlir::Block*> rangeBlocks;

  llvm::DenseMap<MultidimensionalRangeAttr, mlir::Block*>
      multidimensionalRangeBlocks;

  // Create a block for each unique range.
  mlir::Block* firstRangeBlock = nullptr;

  for (const auto& range : uniqueRanges) {
    mlir::Block* block = builder.createBlock(returnBlock);
    rangeBlocks[range] = block;

    if (firstRangeBlock == nullptr) {
      firstRangeBlock = block;
    }

    builder.setInsertionPointToStart(block);

    mlir::Value result = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(boundaryGetterCallback(range)));

    builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
  }

  // Create a block for each unique multidimensional range.
  mlir::Block* firstMultidimensionalRangeBlock = nullptr;

  for (const auto& multidimensionalRange : uniqueMultidimensionalRanges) {
    llvm::SmallVector<int64_t, 3> caseValues;
    llvm::SmallVector<mlir::Block*, 3> caseBlocks;
    llvm::SmallVector<mlir::ValueRange, 3> caseOperandsRefs;

    for (const auto& range :
         llvm::enumerate(multidimensionalRange.getRanges())) {
      caseValues.push_back(range.index());
      caseBlocks.push_back(rangeBlocks[range.value()]);
      caseOperandsRefs.push_back(llvm::None);
    }

    assert(firstRangeBlock != nullptr);

    // The block takes as argument the index of the dimension of interest.
    llvm::SmallVector<mlir::Type, 1> blockArgTypes(1, builder.getI64Type());
    llvm::SmallVector<mlir::Location, 1> blockArgLocations(1, loc);

    mlir::Block* block = builder.createBlock(
        firstRangeBlock, blockArgTypes, blockArgLocations);

    multidimensionalRangeBlocks[multidimensionalRange] = block;

    if (firstMultidimensionalRangeBlock == nullptr) {
      firstMultidimensionalRangeBlock = block;
    }

    builder.setInsertionPointToStart(block);

    // The switch operates on the index of the dimension of interest.
    builder.create<mlir::cf::SwitchOp>(
        loc,
        block->getArgument(0), returnBlock, unknownResult,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);
  }

  // Create a block for each variable and the switch inside the entry block.
  llvm::SmallVector<int64_t> variablesCaseValues;
  llvm::SmallVector<mlir::Block*> variablesCaseBlocks;
  llvm::SmallVector<mlir::ValueRange> variablesCaseOperandsRefs;

  for (const auto& variable : llvm::enumerate(
           simulationModuleOp.getVariables().getAsRange<VariableAttr>())) {
    llvm::ArrayRef<MultidimensionalRangeAttr> ranges =
        variable.value().getPrintableIndices();

    if (ranges.empty()) {
      // Scalar variable.
      continue;
    }

    // Create the block for the variable.
    // The arguments are the index of the multidimensional range and its
    // dimension of interest.
    variablesCaseValues.push_back(variable.index());

    assert(firstMultidimensionalRangeBlock != nullptr);
    llvm::SmallVector<mlir::Type, 2> blockArgTypes(2, builder.getI64Type());
    llvm::SmallVector<mlir::Location, 2> blockArgLocations(2, loc);

    mlir::Block* block = builder.createBlock(
        firstMultidimensionalRangeBlock, blockArgTypes, blockArgLocations);

    variablesCaseBlocks.push_back(block);

    variablesCaseOperandsRefs.push_back(
        entryBlock->getArguments().drop_front());

    // Create the switch for the multidimensional ranges.
    // The switch operates on the index of the multidimensional range.
    llvm::SmallVector<int64_t, 1> caseValues;
    llvm::SmallVector<mlir::Block*, 1> caseBlocks;
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 2>, 1> caseOperands;
    llvm::SmallVector<mlir::ValueRange, 1> caseOperandsRefs;

    for (const auto& multidimensionalRange : llvm::enumerate(ranges)) {
      caseValues.push_back(multidimensionalRange.index());

      caseBlocks.push_back(
          multidimensionalRangeBlocks[multidimensionalRange.value()]);

      caseOperands.resize(1);
      caseOperands[0].push_back(block->getArgument(1));
      caseOperandsRefs.push_back(caseOperands[0]);
    }

    builder.setInsertionPointToStart(block);

    builder.create<mlir::cf::SwitchOp>(
        loc,
        block->getArgument(0), returnBlock, unknownResult,
        builder.getI64TensorAttr(caseValues),
        caseBlocks, caseOperandsRefs);
  }

  // Create the switch inside the entry block.
  // The switch operates on the index of the variable.
  builder.setInsertionPointToEnd(entryBlock);

  builder.create<mlir::cf::SwitchOp>(
      loc,
      entryBlock->getArgument(0), returnBlock, unknownResult,
      builder.getI64TensorAttr(variablesCaseValues),
      variablesCaseBlocks, variablesCaseOperandsRefs);
}

void ModuleOpLowering::createGetDerivativeFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "getDerivative",
      builder.getFunctionType(builder.getI64Type(), builder.getI64Type()));

  // Create the entry block.
  mlir::Block* entryBlock = funcOp.addEntryBlock();

  // Create the last block receiving the value to be returned.
  mlir::Block* returnBlock = builder.createBlock(
      &funcOp.getFunctionBody(),
      funcOp.getFunctionBody().end(),
      builder.getI64Type(),
      loc);

  builder.setInsertionPointToEnd(returnBlock);
  builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

  // Map each variable name to the variable position.
  mlir::ArrayAttr variables = simulationModuleOp.getVariables();
  llvm::StringMap<size_t> variablesPositionsMap;

  for (const auto& variable :
       llvm::enumerate(variables.getAsRange<VariableAttr>())) {
    variablesPositionsMap[variable.value().getName()] = variable.index();
  }

  // Get the map of the derivatives.
  llvm::StringMap<llvm::StringRef> derivativesMap =
      simulationModuleOp.getDerivativesMap();

  // Create the blocks and the switch.
  llvm::SmallVector<int64_t> caseValues;
  llvm::SmallVector<mlir::Block*> caseBlocks;
  llvm::SmallVector<mlir::ValueRange> caseOperandsRefs;

  for (VariableAttr variable : variables.getAsRange<VariableAttr>()) {
    llvm::StringRef variableName = variable.getName();
    auto it = derivativesMap.find(variableName);

    if (it != derivativesMap.end()) {
      llvm::StringRef derivativeName = it->getValue();
      caseValues.push_back(variablesPositionsMap[variableName]);

      mlir::Block* caseBlock = builder.createBlock(returnBlock);
      caseBlocks.push_back(caseBlock);

      caseOperandsRefs.push_back(llvm::None);

      // Populate the block.
      builder.setInsertionPointToStart(caseBlock);
      int64_t derivativeIndex = variablesPositionsMap[derivativeName];

      mlir::Value blockResult = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI64IntegerAttr(derivativeIndex));

      builder.create<mlir::cf::BranchOp>(loc, returnBlock, blockResult);
    }
  }

  // Populate the entry block.
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value defaultOperand = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(-1));

  builder.create<mlir::cf::SwitchOp>(
      loc,
      entryBlock->getArgument(0), returnBlock, defaultOperand,
      builder.getI64TensorAttr(caseValues),
      caseBlocks, caseOperandsRefs);
}

void ModuleOpLowering::createGetTimeFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "getTime",
      builder.getFunctionType(getVoidPtrType(), builder.getF64Type()));

  mlir::Block* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Load the runtime data structure.
  mlir::Value structValue = loadRuntimeDataFromOpaquePtr(
      builder, funcOp.getArgument(0), simulationModuleOp.getVariablesTypes());

  // Extract the time variable.
  mlir::Value time = extractTime(builder, structValue);

  builder.create<mlir::func::ReturnOp>(loc, time);
}

void ModuleOpLowering::createSetTimeFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  llvm::SmallVector<mlir::Type, 2> argTypes;
  argTypes.push_back(getVoidPtrType());
  argTypes.push_back(builder.getF64Type());

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "setTime",
      builder.getFunctionType(argTypes, llvm::None));

  mlir::Block* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto variablesTypes = simulationModuleOp.getVariablesTypes();

  // Load the runtime data structure.
  mlir::Value runtimeData = loadRuntimeDataFromOpaquePtr(
      builder, funcOp.getArgument(0), variablesTypes);

  // Set the time inside the data structure.
  mlir::Value time = funcOp.getArgument(1);

  runtimeData = builder.create<mlir::LLVM::InsertValueOp>(
      loc, runtimeData, time, timeVariablePosition);

  setRuntimeDataIntoOpaquePtr(
      builder, funcOp.getArgument(0), runtimeData, variablesTypes);

  // Terminate the function.
  builder.create<mlir::func::ReturnOp>(loc);
}

void ModuleOpLowering::createGetVariableValueFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Convert the variable getters.
  llvm::DenseMap<VariableAttr, mlir::func::FuncOp> attributesGettersMap;
  size_t numOfGetters = 0;

  simulationModuleOp.walk([&](VariableGetterOp op) {
    mlir::func::FuncOp funcOp = convertVariableGetterOp(
        builder, simulationModuleOp, op, numOfGetters);

    for (mlir::Attribute variable : op.getVariables()) {
      attributesGettersMap[variable.cast<VariableAttr>()] = funcOp;
    }
  });

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Type int64PtrType =
      mlir::LLVM::LLVMPointerType::get(builder.getI64Type());

  llvm::SmallVector<mlir::Type, 3> argTypes;
  argTypes.push_back(getVoidPtrType());
  argTypes.push_back(builder.getI64Type());
  argTypes.push_back(int64PtrType);

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "getVariableValue",
      builder.getFunctionType(argTypes, builder.getF64Type()));

  // Create the entry block.
  mlir::Block* entryBlock = funcOp.addEntryBlock();

  // Load the runtime data structure.
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value runtimeData = loadRuntimeDataFromOpaquePtr(
      builder, funcOp.getArgument(0), simulationModuleOp.getVariablesTypes());

  // Create the last block receiving the value to be returned.
  mlir::Block* returnBlock = builder.createBlock(
      &funcOp.getFunctionBody(),
      funcOp.getFunctionBody().end(),
      builder.getF64Type(),
      loc);

  builder.setInsertionPointToStart(returnBlock);
  builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

  // Create the blocks.
  llvm::SmallVector<int64_t> caseValues;
  llvm::SmallVector<mlir::Block*> caseBlocks;
  llvm::SmallVector<mlir::ValueRange> caseOperandsRefs;

  llvm::SmallVector<mlir::Value, 2> callArgs;
  callArgs.resize(2);

  callArgs[1] = funcOp.getArgument(2);

  for (const auto& variable : llvm::enumerate(
           simulationModuleOp.getVariables().getAsRange<VariableAttr>())) {
    auto it = attributesGettersMap.find(variable.value());

    if (it == attributesGettersMap.end()) {
      // No getter has been provided for the variable.
      continue;
    }

    caseValues.push_back(variable.index());

    mlir::Block* caseBlock = builder.createBlock(returnBlock);
    caseBlocks.push_back(caseBlock);

    builder.setInsertionPointToStart(caseBlock);
    callArgs[0] = extractVariable(builder, runtimeData, variable.index());

    auto callOp = builder.create<mlir::func::CallOp>(
        loc, it->getSecond(), callArgs);

    mlir::Value result = callOp.getResult(0);
    builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);

    caseOperandsRefs.push_back(llvm::None);
  }

  // Populate the entry block.
  builder.setInsertionPointToEnd(entryBlock);

  mlir::Value defaultOperand = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getF64FloatAttr(0));

  builder.create<mlir::cf::SwitchOp>(
      loc,
      entryBlock->getArgument(1), returnBlock, defaultOperand,
      builder.getI64TensorAttr(caseValues),
      caseBlocks, caseOperandsRefs);
}

mlir::func::FuncOp ModuleOpLowering::convertVariableGetterOp(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp,
    VariableGetterOp op,
    size_t& convertedGetters) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Type int64PtrType =
      mlir::LLVM::LLVMPointerType::get(builder.getI64Type());

  llvm::SmallVector<mlir::Type, 3> argTypes;
  argTypes.push_back(op.getVariable().getType());
  argTypes.push_back(int64PtrType);

  auto funcOp = builder.create<mlir::func::FuncOp>(
      loc, "getVariableValue" + std::to_string(convertedGetters++),
      builder.getFunctionType(argTypes, builder.getF64Type()));

  mlir::BlockAndValueMapping mapping;

  // Create the entry block.
  mlir::Block* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Map the variable.
  mapping.map(op.getVariable(), funcOp.getArgument(0));

  // Extract and map the indices.
  for (int64_t i = 0, e = op.getVariableRank(); i < e; ++i) {
    mlir::Value offset = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(i));

    mlir::Value address = builder.create<mlir::LLVM::GEPOp>(
        loc, int64PtrType, int64PtrType, funcOp.getArgument(1), offset);

    mlir::Value index = builder.create<mlir::LLVM::LoadOp>(loc, address);

    index = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getIndexType(), index);

    mapping.map(op.getIndex(i), index);
  }

  // Clone the blocks structure.
  for (auto& block : llvm::enumerate(op.getBodyRegion())) {
    if (block.index() == 0) {
      mapping.map(&block.value(), entryBlock);
    } else {
      std::vector<mlir::Location> argLocations;

      for (const auto& arg : block.value().getArguments()) {
        argLocations.push_back(arg.getLoc());
      }

      mlir::Block* clonedBlock = builder.createBlock(
          &funcOp.getFunctionBody(),
          funcOp.getFunctionBody().end(),
          block.value().getArgumentTypes(),
          argLocations);

      mapping.map(&block.value(), clonedBlock);

      // Map the block arguments.
      for (const auto& [original, cloned] : llvm::zip(
               block.value().getArguments(), clonedBlock->getArguments())) {
        mapping.map(original, cloned);
      }
    }
  }

  // Clone the operations.
  for (auto& block : llvm::enumerate(op.getBodyRegion().getBlocks())) {
    mlir::Block* clonedBlock = mapping.lookup(&block.value());
    builder.setInsertionPointToEnd(clonedBlock);

    for (auto& bodyOp : block.value().getOperations()) {
      if (auto yieldOp = mlir::dyn_cast<YieldOp>(bodyOp)) {
        std::vector<mlir::Value> returnedValues;

        for (mlir::Value yieldedValue : yieldOp.getValues()) {
          mlir::Value returnedValue = mapping.lookup(yieldedValue);

          if (returnedValue.getType().isa<mlir::IndexType>()) {
            returnedValue = builder.create<mlir::arith::IndexCastOp>(
                returnedValue.getLoc(), builder.getI64Type(), returnedValue);
          }

          if (returnedValue.getType().isa<mlir::IntegerType>()) {
            if (returnedValue.getType().getIntOrFloatBitWidth() < 64) {
              returnedValue = builder.create<mlir::arith::ExtSIOp>(
                  returnedValue.getLoc(), builder.getI64Type(), returnedValue);
            } else if (returnedValue.getType().getIntOrFloatBitWidth() > 64) {
              returnedValue = builder.create<mlir::arith::TruncIOp>(
                  returnedValue.getLoc(), builder.getF64Type(), returnedValue);
            }

            returnedValue = builder.create<mlir::arith::SIToFPOp>(
                returnedValue.getLoc(), builder.getF64Type(), returnedValue);
          }

          if (returnedValue.getType().getIntOrFloatBitWidth() < 64) {
            returnedValue = builder.create<mlir::LLVM::FPExtOp>(
                loc, builder.getF64Type(), returnedValue);
          } else if (returnedValue.getType().getIntOrFloatBitWidth() > 64) {
            returnedValue = builder.create<mlir::LLVM::FPTruncOp>(
                loc, builder.getF64Type(), returnedValue);
          }

          returnedValues.push_back(returnedValue);
        }

        builder.create<mlir::func::ReturnOp>(yieldOp.getLoc(), returnedValues);
      } else {
        builder.clone(bodyOp, mapping);
      }
    }
  }

  return funcOp;
}

void ModuleOpLowering::createMainFunction(
    mlir::OpBuilder& builder,
    mlir::simulation::ModuleOp simulationModuleOp) const
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = simulationModuleOp.getLoc();

  // Create the function inside the parent module.
  auto moduleOp = simulationModuleOp->getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  mlir::Type resultType = builder.getI32Type();

  llvm::SmallVector<mlir::Type, 3> argTypes;
  argTypes.push_back(builder.getI32Type());

  mlir::Type charType = builder.getIntegerType(8);
  mlir::Type charPtrType = mlir::LLVM::LLVMPointerType::get(charType);

  mlir::Type charPtrPtrType =
      mlir::LLVM::LLVMPointerType::get(charPtrType);

  argTypes.push_back(charPtrPtrType);

  auto mainFunction = builder.create<mlir::func::FuncOp>(
      loc, "main",
      builder.getFunctionType(argTypes, resultType));

  mlir::Block* entryBlock = mainFunction.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Call the function to start the simulation.
  // Its definition lives within the runtime library.

  auto runFunction = declareExternalFunction(
      builder, moduleOp, "runSimulation",
      mlir::LLVM::LLVMFunctionType::get(resultType, argTypes));

  auto runSimulationCall = builder.create<mlir::LLVM::CallOp>(
      loc, runFunction, mainFunction.getArguments());

  mlir::Value returnValue = runSimulationCall.getResult();

  // Create the return statement.
  builder.create<mlir::func::ReturnOp>(loc, returnValue);
}

namespace
{
  class SimulationToFuncConversionPass
      : public mlir::impl::SimulationToFuncConversionPassBase<
          SimulationToFuncConversionPass>
  {
    public:
      using SimulationToFuncConversionPassBase
        ::SimulationToFuncConversionPassBase;

      void runOnOperation() override
      {
        llvm::SmallVector<mlir::simulation::ModuleOp, 1> moduleOps;

        getOperation().walk([&](mlir::simulation::ModuleOp op) {
          moduleOps.push_back(op);
        });

        for (mlir::simulation::ModuleOp moduleOp : moduleOps) {
          if (mlir::failed(convertFunctionLikeOps(moduleOp))) {
            return signalPassFailure();
          }

          if (mlir::failed(convertModuleOp(moduleOp))) {
            return signalPassFailure();
          }
        }
      }

    private:
      mlir::LogicalResult convertFunctionLikeOps(
        mlir::simulation::ModuleOp simulationModuleOp);

      mlir::LogicalResult convertModuleOp(
          mlir::simulation::ModuleOp simulationModuleOp);
  };
}

mlir::LogicalResult SimulationToFuncConversionPass::convertFunctionLikeOps(
    mlir::simulation::ModuleOp simulationModuleOp)
{
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<mlir::LLVM::LLVMDialect>();

  target.addIllegalOp<
      InitFunctionOp,
      DeinitFunctionOp,
      InitICSolversFunctionOp,
      InitMainSolversFunctionOp,
      DeinitICSolversFunctionOp,
      DeinitMainSolversFunctionOp,
      FunctionOp,
      ReturnOp>();

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());

  populateSimulationToFuncFunctionLikeConversionPatterns(
      patterns, &getContext());

  return applyPartialConversion(
      simulationModuleOp, target, std::move(patterns));
}

mlir::LogicalResult SimulationToFuncConversionPass::convertModuleOp(
    mlir::simulation::ModuleOp simulationModuleOp)
{
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalOp<mlir::simulation::ModuleOp>();

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<ModuleOpLowering>(&getContext(), emitMainFunction);

  return applyPartialConversion(
      simulationModuleOp, target, std::move(patterns));
}

static VariableAttr convertVariableType(
    VariableAttr variable, mlir::TypeConverter* typeConverter)
{
  return VariableAttr::get(
      variable.getContext(),
      typeConverter->convertType(variable.getType()),
      variable.getName(),
      variable.getDimensions(),
      variable.getPrintable(),
      variable.getPrintableIndices());
}

namespace
{
  struct ModuleOpTypes
      : public mlir::OpConversionPattern<mlir::simulation::ModuleOp>
  {
    using mlir::OpConversionPattern<mlir::simulation::ModuleOp>
        ::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        mlir::simulation::ModuleOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto newOp = mlir::cast<mlir::simulation::ModuleOp>(
          rewriter.clone(*op.getOperation()));

      // Compute and set the variables with converted types.
      llvm::SmallVector<mlir::Attribute> variables;

      for (VariableAttr variable :
           op.getVariables().getAsRange<VariableAttr>()) {
        variables.push_back(convertVariableType(variable, getTypeConverter()));
      }

      newOp->setAttr(
          op.getVariablesAttrName(),
          rewriter.getArrayAttr(variables));

      // Replace the old module.
      rewriter.replaceOp(op, newOp->getResults());

      return mlir::success();
    }
  };

  template<typename Op>
  struct FunctionLikeOpTypes : public mlir::ConvertOpToLLVMPattern<Op>
  {
    using mlir::ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

    mlir::LogicalResult cloneBody(
        mlir::OpBuilder& builder,
        mlir::BlockAndValueMapping& mapping,
        mlir::Region& source,
        mlir::Region& destination) const
    {
      assert(destination.getBlocks().size() == 1);
      mlir::Block* entryBlock = &destination.front();

      // Clone the blocks structure.
      for (auto& block : llvm::enumerate(source)) {
        if (block.index() == 0) {
          mapping.map(&block.value(), entryBlock);
        } else {
          std::vector<mlir::Location> argLocations;

          for (const auto& arg : block.value().getArguments()) {
            argLocations.push_back(arg.getLoc());
          }

          auto signatureConversion =
              this->getTypeConverter()->convertBlockSignature(&block.value());

          if (!signatureConversion) {
            return mlir::failure();
          }

          mlir::Block* clonedBlock = builder.createBlock(
              &destination,
              destination.end(),
              signatureConversion->getConvertedTypes(),
              argLocations);

          mapping.map(&block.value(), clonedBlock);
        }
      }

      // Map the block arguments.
      for (auto& block : llvm::enumerate(source.getBlocks())) {
        mlir::Block* clonedBlock = mapping.lookup(&block.value());
        builder.setInsertionPointToEnd(clonedBlock);

        // Cast the block arguments.
        for (const auto& [original, cloned] : llvm::zip(
                 block.value().getArguments(), clonedBlock->getArguments())) {
          mlir::Value mappedArg = cloned;

          if (block.index() == 0) {
            if (mappedArg.getType() != original.getType()) {
              mappedArg =
                  this->getTypeConverter()->materializeSourceConversion(
                      builder, cloned.getLoc(), original.getType(), mappedArg);
            }
          }

          mapping.map(original, mappedArg);
        }

        // Clone the operations.
        for (auto& bodyOp : block.value().getOperations()) {
          mlir::Operation* clonedBodyOp = builder.clone(bodyOp, mapping);

          if (bodyOp.hasTrait<mlir::OpTrait::IsTerminator>()) {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(clonedBodyOp);
            llvm::SmallVector<mlir::Value, 3> castedOperands;

            for (mlir::Value operand : clonedBodyOp->getOperands()) {
              mlir::Value newOperand = operand;

              if (!this->getTypeConverter()->isLegal(newOperand.getType())) {
                newOperand = this->getTypeConverter()->materializeTargetConversion(
                    builder, newOperand.getLoc(),
                    this->getTypeConverter()->convertType(newOperand.getType()),
                    newOperand);
              }

              castedOperands.push_back(newOperand);
            }

            clonedBodyOp->setOperands(castedOperands);
          }
        }
      }

      return mlir::success();
    }
  };

  struct InitFunctionOpTypes : public FunctionLikeOpTypes<InitFunctionOp>
  {
    using FunctionLikeOpTypes<InitFunctionOp>::FunctionLikeOpTypes;

    mlir::LogicalResult matchAndRewrite(
        InitFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      // Convert the signature.
      mlir::TypeConverter::SignatureConversion signatureConversion(
          op.getFunctionType().getNumInputs());

      auto functionType =
          this->getTypeConverter()->convertFunctionSignature(
                                      op.getFunctionType(),
                                      false, signatureConversion)
              .template cast<mlir::LLVM::LLVMFunctionType>();

      auto newOp = rewriter.replaceOpWithNewOp<InitFunctionOp>(
          op,
          rewriter.getFunctionType(
              functionType.getParams(), functionType.getReturnType()));

      newOp.addEntryBlock();
      mlir::BlockAndValueMapping mapping;

      return cloneBody(
          rewriter, mapping, op.getBodyRegion(), newOp.getBodyRegion());
    }
  };

  struct DeinitFunctionOpTypes : public FunctionLikeOpTypes<DeinitFunctionOp>
  {
    using FunctionLikeOpTypes<DeinitFunctionOp>::FunctionLikeOpTypes;

    mlir::LogicalResult matchAndRewrite(
        DeinitFunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      // Convert the signature.
      mlir::TypeConverter::SignatureConversion signatureConversion(
          op.getFunctionType().getNumInputs());

      auto functionType =
          this->getTypeConverter()->convertFunctionSignature(
                                      op.getFunctionType(),
                                      false, signatureConversion)
              .template cast<mlir::LLVM::LLVMFunctionType>();

      auto newOp = rewriter.replaceOpWithNewOp<DeinitFunctionOp>(
          op,
          rewriter.getFunctionType(
              functionType.getParams(), functionType.getReturnType()));

      newOp.addEntryBlock();
      mlir::BlockAndValueMapping mapping;

      return cloneBody(
          rewriter, mapping, op.getBodyRegion(), newOp.getBodyRegion());
    }
  };

  template<typename Op>
  struct InitSolversFunctionOpTypes : public FunctionLikeOpTypes<Op>
  {
    using FunctionLikeOpTypes<Op>::FunctionLikeOpTypes;
    using OpAdaptor = typename FunctionLikeOpTypes<Op>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(
        Op op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Type> solverTypes;
      llvm::SmallVector<mlir::Type> variableTypes;

      for (mlir::Type solverType : op.getSolverTypes()) {
        solverTypes.push_back(
            this->getTypeConverter()->convertType(solverType));
      }

      for (mlir::Type variableType : op.getVariableTypes()) {
        variableTypes.push_back(
            this->getTypeConverter()->convertType(variableType));
      }

      auto newOp = rewriter.replaceOpWithNewOp<Op>(
          op, solverTypes, variableTypes);

      for (mlir::NamedAttribute attr : op->getAttrs()) {
        if (attr.getName() == op.getFunctionTypeAttrName()) {
          continue;
        }

        newOp->setAttr(attr.getName(), attr.getValue());
      }

      newOp.addEntryBlock();

      mlir::BlockAndValueMapping mapping;

      return this->cloneBody(
          rewriter, mapping, op.getBodyRegion(), newOp.getBodyRegion());
    }
  };

  struct InitICSolversFunctionOpTypes
      : public InitSolversFunctionOpTypes<InitICSolversFunctionOp>
  {
    using InitSolversFunctionOpTypes<InitICSolversFunctionOp>
        ::InitSolversFunctionOpTypes;
  };

  struct InitMainSolversFunctionOpTypes
      : public InitSolversFunctionOpTypes<InitMainSolversFunctionOp>
  {
    using InitSolversFunctionOpTypes<InitMainSolversFunctionOp>
        ::InitSolversFunctionOpTypes;
  };

  template<typename Op>
  struct DeinitSolversFunctionOpTypes : public FunctionLikeOpTypes<Op>
  {
    using FunctionLikeOpTypes<Op>::FunctionLikeOpTypes;
    using OpAdaptor = typename FunctionLikeOpTypes<Op>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(
        Op op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      // Convert the signature.
      mlir::TypeConverter::SignatureConversion signatureConversion(
          op.getFunctionType().getNumInputs());

      auto functionType =
          this->getTypeConverter()->convertFunctionSignature(
                                op.getFunctionType(),
                                false, signatureConversion)
              .template cast<mlir::LLVM::LLVMFunctionType>();

      auto newOp = rewriter.replaceOpWithNewOp<Op>(
          op,
          rewriter.getFunctionType(
              functionType.getParams(), functionType.getReturnType()));

      newOp.addEntryBlock();
      mlir::BlockAndValueMapping mapping;

      return this->cloneBody(
          rewriter, mapping, op.getBodyRegion(), newOp.getBodyRegion());
    }
  };

  struct DeinitICSolversFunctionOpTypes
      : public DeinitSolversFunctionOpTypes<DeinitICSolversFunctionOp>
  {
    using DeinitSolversFunctionOpTypes<DeinitICSolversFunctionOp>
        ::DeinitSolversFunctionOpTypes;
  };

  struct DeinitMainSolversFunctionOpTypes
      : public DeinitSolversFunctionOpTypes<DeinitMainSolversFunctionOp>
  {
    using DeinitSolversFunctionOpTypes<DeinitMainSolversFunctionOp>
        ::DeinitSolversFunctionOpTypes;
  };

  struct VariableGetterOpTypes : public FunctionLikeOpTypes<VariableGetterOp>
  {
    using FunctionLikeOpTypes<VariableGetterOp>::FunctionLikeOpTypes;

    mlir::LogicalResult matchAndRewrite(
        VariableGetterOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      // Convert the signature.
      mlir::TypeConverter::SignatureConversion signatureConversion(
          op.getFunctionType().getNumInputs());

      auto functionType =
          getTypeConverter()->convertFunctionSignature(
                                op.getFunctionType(),
                                false, signatureConversion)
              .cast<mlir::LLVM::LLVMFunctionType>();

      // Convert the types of the variables.
      llvm::SmallVector<mlir::Attribute> variables;

      for (VariableAttr variable :
           op.getVariables().getAsRange<VariableAttr>()) {
        variables.push_back(convertVariableType(variable, getTypeConverter()));
      }

      // Create the new operation.
      auto newOp = rewriter.replaceOpWithNewOp<VariableGetterOp>(
          op,
          rewriter.getFunctionType(
              functionType.getParams(), functionType.getReturnType()),
          rewriter.getArrayAttr(variables));

      // Populate the body.
      newOp.addEntryBlock();
      mlir::BlockAndValueMapping mapping;

      return cloneBody(
          rewriter, mapping, op.getBodyRegion(), newOp.getBodyRegion());
    }
  };

  struct FunctionOpTypes : public FunctionLikeOpTypes<FunctionOp>
  {
    using FunctionLikeOpTypes<FunctionOp>::FunctionLikeOpTypes;

    mlir::LogicalResult matchAndRewrite(
        FunctionOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Type> solverTypes;
      mlir::Type timeType;
      llvm::SmallVector<mlir::Type> variableTypes;
      llvm::SmallVector<mlir::Type> extraArgTypes;
      llvm::SmallVector<mlir::Type> resultTypes;

      for (mlir::Type type : op.getSolverTypes()) {
        solverTypes.push_back(getTypeConverter()->convertType(type));
      }

      timeType = getTypeConverter()->convertType(op.getTimeType());

      for (mlir::Type type : op.getVariableTypes()) {
        variableTypes.push_back(getTypeConverter()->convertType(type));
      }

      for (mlir::Type type : op.getExtraArgTypes()) {
        extraArgTypes.push_back(getTypeConverter()->convertType(type));
      }

      for (mlir::Type type : op.getResultTypes()) {
        resultTypes.push_back(getTypeConverter()->convertType(type));
      }

      auto newOp = rewriter.replaceOpWithNewOp<FunctionOp>(
          op,
          op.getSymName(),
          solverTypes,
          timeType,
          variableTypes,
          extraArgTypes,
          resultTypes);

      for (mlir::NamedAttribute attr : op->getAttrs()) {
        if (attr.getName() == op.getFunctionTypeAttrName()) {
          continue;
        }

        newOp->setAttr(attr.getName(), attr.getValue());
      }

      newOp.addEntryBlock();

      mlir::BlockAndValueMapping mapping;

      return cloneBody(
          rewriter, mapping, op.getBodyRegion(), newOp.getBodyRegion());
    }
  };
}

namespace mlir
{
  void populateSimulationToFuncStructuralTypeConversionsAndLegality(
      mlir::LLVMTypeConverter& typeConverter,
      mlir::RewritePatternSet& patterns,
      mlir::ConversionTarget& target)
  {
    patterns.add<ModuleOpTypes>(typeConverter, patterns.getContext());

    patterns.add<
        InitFunctionOpTypes,
        DeinitFunctionOpTypes,
        InitICSolversFunctionOpTypes,
        InitMainSolversFunctionOpTypes,
        DeinitICSolversFunctionOpTypes,
        DeinitMainSolversFunctionOpTypes,
        VariableGetterOpTypes,
        FunctionOpTypes>(typeConverter);

    target.addDynamicallyLegalOp<mlir::simulation::ModuleOp>(
        [&typeConverter](mlir::simulation::ModuleOp op) {
          return llvm::all_of(
              op.getVariables().getAsRange<VariableAttr>(),
                  [&](VariableAttr attr) {
                return typeConverter.isLegal(attr.getType());
              });
        });

    target.addDynamicallyLegalOp<InitFunctionOp>(
        [&typeConverter](InitFunctionOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
              typeConverter.isLegal(&op.getBodyRegion());
        });

    target.addDynamicallyLegalOp<DeinitFunctionOp>(
        [&typeConverter](DeinitFunctionOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
              typeConverter.isLegal(&op.getBodyRegion());
        });

    target.addDynamicallyLegalOp<InitICSolversFunctionOp>(
        [&](InitICSolversFunctionOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
              typeConverter.isLegal(&op.getBodyRegion());
        });

    target.addDynamicallyLegalOp<InitMainSolversFunctionOp>(
        [&](InitMainSolversFunctionOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
              typeConverter.isLegal(&op.getBodyRegion());
        });

    target.addDynamicallyLegalOp<DeinitICSolversFunctionOp>(
        [&](DeinitICSolversFunctionOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
              typeConverter.isLegal(&op.getBodyRegion());
        });

    target.addDynamicallyLegalOp<DeinitMainSolversFunctionOp>(
        [&](DeinitMainSolversFunctionOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
              typeConverter.isLegal(&op.getBodyRegion());
        });

    target.addDynamicallyLegalOp<VariableGetterOp>(
        [&typeConverter](VariableGetterOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
              typeConverter.isLegal(&op.getBodyRegion());
        });

    target.addDynamicallyLegalOp<FunctionOp>(
        [&typeConverter](FunctionOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
              typeConverter.isLegal(&op.getBodyRegion());
        });
  }

  std::unique_ptr<mlir::Pass> createSimulationToFuncConversionPass()
  {
    return std::make_unique<SimulationToFuncConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createSimulationToFuncConversionPass(
      const SimulationToFuncConversionPassOptions& options)
  {
    return std::make_unique<SimulationToFuncConversionPass>(options);
  }
}
