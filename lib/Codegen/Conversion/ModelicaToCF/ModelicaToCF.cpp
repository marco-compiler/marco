#include "marco/Codegen/Conversion/ModelicaToCF/ModelicaToCF.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include <set>
#include <stack>

namespace mlir
{
#define GEN_PASS_DEF_MODELICATOCFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::modelica;

/// Remove the unreachable blocks of a region.
static void removeUnreachableBlocks(mlir::Region& region)
{
  std::stack<mlir::Block*> unreachableBlocks;

  auto collectUnreachableBlocks = [&]() {
    for (auto& block : region.getBlocks()) {
      if (block.hasNoPredecessors() && !block.isEntryBlock()) {
        unreachableBlocks.push(&block);
      }
    }
  };

  collectUnreachableBlocks();

  do {
    while (!unreachableBlocks.empty()) {
      unreachableBlocks.top()->erase();
      unreachableBlocks.pop();
    }

    // Erasing some blocks may have turned their successors into unreachable
    // blocks.
    collectUnreachableBlocks();
  } while (!unreachableBlocks.empty());
}

using LoadReplacer = std::function<mlir::LogicalResult(mlir::OpBuilder&, MemberLoadOp)>;
using StoreReplacer = std::function<mlir::LogicalResult(mlir::OpBuilder&, MemberStoreOp)>;

/// Convert a member that is provided as input to the function.
/// The replacement is the argument of the raw function.
static mlir::LogicalResult convertArgument(
    mlir::OpBuilder& builder, MemberCreateOp op, mlir::Value replacement)
{
  auto unwrappedType = op.getMemberType().unwrap();

  auto replacers = [&]() {
    if (!unwrappedType.isa<ArrayType>()) {
      // The value is a scalar.
      assert(op.getDynamicSizes().empty());

      return std::make_pair<LoadReplacer, StoreReplacer>(
          [=](mlir::OpBuilder& builder, MemberLoadOp loadOp)
              -> mlir::LogicalResult {
            loadOp.replaceAllUsesWith(replacement);
            loadOp.erase();
            return mlir::success();
          },
          [](mlir::OpBuilder& builder, MemberStoreOp storeOp)
              -> mlir::LogicalResult {
            llvm_unreachable("Store on input scalar argument");
            return mlir::failure();
          });
    }

    // Only true input members are allowed to have dynamic dimensions.
    // The output values that have been promoted to input arguments must have
    // a static shape in order to cover possible reassignments.
    assert(op.isInput() || op.getMemberType().toArrayType().hasStaticShape());

    return std::make_pair<LoadReplacer, StoreReplacer>(
        [=](mlir::OpBuilder& builder, MemberLoadOp loadOp)
            -> mlir::LogicalResult {
          loadOp.replaceAllUsesWith(replacement);
          loadOp.erase();
          return mlir::success();
        },
        [=](mlir::OpBuilder& builder, MemberStoreOp storeOp)
            -> mlir::LogicalResult {
          builder.setInsertionPoint(storeOp);

          builder.create<ArrayCopyOp>(
              storeOp.getLoc(), storeOp.getValue(), replacement);

          storeOp->erase();
          return mlir::success();
        });
  };

  LoadReplacer loadReplacer;
  StoreReplacer storeReplacer;
  std::tie(loadReplacer, storeReplacer) = replacers();

  for (auto* user : llvm::make_early_inc_range(op->getUsers())) {
    assert(mlir::isa<MemberLoadOp>(user) || mlir::isa<MemberStoreOp>(user));

    if (auto loadOp = mlir::dyn_cast<MemberLoadOp>(user)) {
      if (mlir::failed(loadReplacer(builder, loadOp))) {
        return mlir::failure();
      }
    } else if (auto storeOp = mlir::dyn_cast<MemberStoreOp>(user)) {
      if (mlir::failed(storeReplacer(builder, storeOp))) {
        return mlir::failure();
      }
    }
  }

  op->erase();
  return mlir::success();
}

static ArrayType getArrayTypeWithDynamicDimensionsSetToZero(
    ArrayType type)
{
  if (!type.hasRank()) {
    return type;
  }

  llvm::SmallVector<int64_t, 3> shape;

  for (int64_t dimension : type.getShape()) {
    if (dimension == mlir::ShapedType::kDynamicSize) {
      shape.push_back(0);
    } else {
      shape.push_back(dimension);
    }
  }

  return ArrayType::get(
      shape, type.getElementType(), type.getMemorySpace());
}

static mlir::LogicalResult convertResultOrProtectedVar(
    mlir::OpBuilder& builder,
    MemberCreateOp op,
    mlir::TypeConverter* typeConverter)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = op.getLoc();
  auto memberType = op.getMemberType();
  auto unwrappedType = memberType.unwrap();

  auto replacers = [&]() {
    if (!unwrappedType.isa<ArrayType>()) {
      // The value is a scalar.
      assert(op.getDynamicSizes().empty());

      builder.setInsertionPoint(op);

      mlir::Value reference = builder.create<AllocaOp>(
          loc, ArrayType::get(llvm::None, unwrappedType), llvm::None);

      return std::make_pair<LoadReplacer, StoreReplacer>(
          [=](mlir::OpBuilder& builder, MemberLoadOp loadOp)
              -> mlir::LogicalResult {
            builder.setInsertionPoint(loadOp);

            mlir::Value replacement = builder.create<LoadOp>(
                loadOp.getLoc(), reference, llvm::None);

            loadOp.replaceAllUsesWith(replacement);
            loadOp.erase();
            return mlir::success();
          },
          [=](mlir::OpBuilder& builder, MemberStoreOp storeOp)
              -> mlir::LogicalResult {
            builder.setInsertionPoint(storeOp);
            mlir::Value value = storeOp.getValue();

            if (value.getType() != unwrappedType) {
              value = builder.create<CastOp>(
                  storeOp.getLoc(), unwrappedType, storeOp.getValue());
            }

            builder.create<StoreOp>(
                storeOp.getLoc(), value, reference, llvm::None);

            storeOp.erase();
            return mlir::success();
          });
    }

    // If we are in the array case, then it may be not sufficient to
    // allocate just the buffer. Instead, if the array has dynamic sizes
    // and those are not initialized, then we need to also allocate a
    // pointer to that buffer, so that we can eventually reassign it if
    // the dimensions change.

    auto arrayType = unwrappedType.cast<ArrayType>();

    bool hasStaticSize = op.getDynamicSizes().size() ==
        static_cast<size_t>(arrayType.getNumDynamicDims());

    if (hasStaticSize) {
      builder.setInsertionPoint(op);

      mlir::Value reference =
          builder.create<AllocOp>(loc, arrayType, op.getDynamicSizes());

      return std::make_pair<LoadReplacer, StoreReplacer>(
          [=](mlir::OpBuilder& builder, MemberLoadOp loadOp)
              -> mlir::LogicalResult {
            loadOp.replaceAllUsesWith(reference);
            loadOp->erase();
            return mlir::success();
          },
          [=](mlir::OpBuilder& builder, MemberStoreOp storeOp)
              -> mlir::LogicalResult {
            builder.setInsertionPoint(storeOp);

            builder.create<ArrayCopyOp>(
                storeOp.getLoc(), storeOp.getValue(), reference);

            storeOp->erase();
            return mlir::success();
          });
    }

    // The array can change sizes during at runtime. Thus, we need to create
    // a pointer to the array currently in use.

    assert(op.getDynamicSizes().empty());
    builder.setInsertionPoint(op);

    // Create the pointer to the array.
    auto memrefOfArrayType = mlir::MemRefType::get(
        llvm::None,
        typeConverter->convertType(memberType.toArrayType()));

    mlir::Value memrefOfArray =
        builder.create<mlir::memref::AllocaOp>(loc, memrefOfArrayType);

    // We need to allocate a fake buffer in order to allow the first free
    // operation to operate on a valid memory area.

    mlir::Value fakeArray = builder.create<AllocOp>(
        loc,
        getArrayTypeWithDynamicDimensionsSetToZero(arrayType),
        llvm::None);

    fakeArray = typeConverter->materializeTargetConversion(
        builder, loc,
        typeConverter->convertType(fakeArray.getType()), fakeArray);

    fakeArray = builder.create<mlir::memref::CastOp>(
        loc, memrefOfArrayType.getElementType(), fakeArray);

    builder.create<mlir::memref::StoreOp>(loc, fakeArray, memrefOfArray);

    return std::make_pair<LoadReplacer, StoreReplacer>(
        [=](mlir::OpBuilder& builder, MemberLoadOp loadOp)
            -> mlir::LogicalResult {
          builder.setInsertionPoint(loadOp);
          mlir::Value array = builder.create<mlir::memref::LoadOp>(
              loadOp.getLoc(), memrefOfArray);

          array = typeConverter->materializeSourceConversion(
              builder, loadOp.getLoc(),
              loadOp.getMemberType().toArrayType(), array);

          loadOp.replaceAllUsesWith(array);
          loadOp->erase();
          return mlir::success();
        },
        [=](mlir::OpBuilder& builder, MemberStoreOp storeOp)
            -> mlir::LogicalResult {
          builder.setInsertionPoint(storeOp);

          // The destination array has dynamic and unknown sizes. Thus, the
          // array has not been allocated yet, and we need to create a copy
          // of the source one.

          mlir::Value value = storeOp.getValue();

          // The function input arguments must be cloned, in order to avoid
          // inputs modifications.

          if (value.isa<mlir::BlockArgument>()) {
            llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

            for (const auto& dimension :
                 llvm::enumerate(arrayType.getShape())) {
              if (dimension.value() == ArrayType::kDynamicSize) {
                mlir::Value dimensionIndex =
                    builder.create<mlir::arith::ConstantOp>(
                        storeOp.getLoc(),
                        builder.getIndexAttr(dimension.index()));

                dynamicDimensions.push_back(builder.create<DimOp>(
                    storeOp.getLoc(), storeOp.getValue(), dimensionIndex));
              }
            }

            value = builder.create<AllocOp>(
                storeOp.getLoc(), arrayType, dynamicDimensions);

            builder.create<ArrayCopyOp>(
                storeOp.getLoc(), storeOp.getValue(), value);
          }

          // Deallocate the previously allocated memory. This is only
          // apparently in contrast with the above statements: unknown-sized
          // arrays pointers are initialized with a pointer to a 1-element
          // sized array, so that the initial free always operates on valid
          // memory.

          mlir::Value previousArray = builder.create<mlir::memref::LoadOp>(
              storeOp.getLoc(), memrefOfArray);

          previousArray = typeConverter->materializeSourceConversion(
              builder, storeOp.getLoc(),
              memberType.toArrayType(), previousArray);

          builder.create<FreeOp>(storeOp.getLoc(), previousArray);

          // Save the descriptor of the new copy into the destination using
          // StoreOp.
          value = typeConverter->materializeTargetConversion(
              builder, storeOp.getLoc(),
              typeConverter->convertType(value.getType()), value);

          value = builder.create<mlir::memref::CastOp>(
              storeOp.getLoc(),
              memrefOfArrayType.getElementType(),
              value);

          builder.create<mlir::memref::StoreOp>(
              storeOp.getLoc(), value, memrefOfArray);

          storeOp->erase();
          return mlir::success();
        });
  };

  LoadReplacer loadReplacer;
  StoreReplacer storeReplacer;
  std::tie(loadReplacer, storeReplacer) = replacers();

  for (auto* user : llvm::make_early_inc_range(op->getUsers())) {
    assert(mlir::isa<MemberLoadOp>(user) || mlir::isa<MemberStoreOp>(user));

    if (auto loadOp = mlir::dyn_cast<MemberLoadOp>(user)) {
      if (mlir::failed(loadReplacer(builder, loadOp))) {
        return mlir::failure();
      }
    } else if (auto storeOp = mlir::dyn_cast<MemberStoreOp>(user)) {
      if (mlir::failed(storeReplacer(builder, storeOp))) {
        return mlir::failure();
      }
    }
  }

  op->erase();
  return mlir::success();
}

static mlir::LogicalResult convertCall(
    mlir::OpBuilder& builder,
    CallOp callOp,
    std::set<size_t> promotedResults)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(callOp);

  llvm::SmallVector<mlir::Value, 3> args;
  llvm::SmallVector<mlir::Type, 3> resultTypes;
  mlir::BlockAndValueMapping mapping;

  for (const auto& resultType : llvm::enumerate(callOp->getResultTypes())) {
    if (promotedResults.find(resultType.index()) == promotedResults.end()) {
      resultTypes.push_back(resultType.value());
    } else {
      // The result has been promoted to argument.
      assert(resultType.value().isa<ArrayType>());
      auto resultArrayType = resultType.value().cast<ArrayType>();
      assert(resultArrayType.hasStaticShape());

      // Allocate the array inside the caller body.
      mlir::Value array = builder.create<AllocOp>(
          callOp.getLoc(), resultArrayType, llvm::None);

      // Add the array to the arguments.
      args.push_back(array);

      // Map the previous result to the array allocated by the caller.
      mapping.map(callOp->getResult(resultType.index()), array);
    }
  }

  for (const auto& arg : callOp.getArgs()) {
    args.push_back(arg);
  }

  auto newCallOp = builder.create<CallOp>(
      callOp.getLoc(), callOp.getCallee(), resultTypes, args);

  size_t newResultsCounter = 0;

  for (const auto& originalResult : callOp.getResults()) {
    mlir::Value mappedResult = mapping.lookupOrNull(originalResult);

    if (mappedResult == nullptr) {
      mlir::Value replacement = newCallOp.getResult(newResultsCounter++);
      originalResult.replaceAllUsesWith(replacement);
    } else {
      originalResult.replaceAllUsesWith(mappedResult);
    }
  }

  callOp->erase();
  return mlir::success();
}

static mlir::LogicalResult convertToRawFunction(
    mlir::OpBuilder& builder,
    FunctionOp functionOp,
    mlir::TypeConverter* typeConverter,
    bool outputArraysPromotion)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(functionOp);

  // Determine which results can be promoted to input arguments.
  std::set<size_t> promotedResults;

  if (outputArraysPromotion) {
    for (const auto& type : llvm::enumerate(functionOp.getResultTypes())) {
      if (auto arrayType = type.value().dyn_cast<ArrayType>();
          arrayType && arrayType.canBeOnStack()) {
        promotedResults.insert(type.index());
      }
    }
  }

  // Determine the function type, taking into account the promoted results.
  llvm::SmallVector<mlir::Type, 3> argTypes;
  llvm::SmallVector<mlir::Type, 3> resultTypes;

  for (const auto& type : llvm::enumerate(functionOp.getResultTypes())) {
    if (promotedResults.find(type.index()) != promotedResults.end()) {
      argTypes.push_back(type.value());
    } else {
      resultTypes.push_back(type.value());
    }
  }

  for (const auto& type : functionOp.getArgumentTypes()) {
    argTypes.push_back(type);
  }

  auto functionType = builder.getFunctionType(argTypes, resultTypes);

  // Create the converted function.
  auto rawFunctionOp = builder.create<RawFunctionOp>(
      functionOp.getLoc(), functionOp.getSymName(), functionType);

  mlir::Block* entryBlock = rawFunctionOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  mlir::BlockAndValueMapping mapping;

  // Clone the blocks structure.
  for (auto& block : llvm::enumerate(functionOp.getBody())) {
    if (block.index() == 0) {
      mapping.map(&block.value(), entryBlock);

    } else {
      std::vector<mlir::Location> argLocations;

      for (const auto& arg : block.value().getArguments()) {
        argLocations.push_back(arg.getLoc());
      }

      mlir::Block* clonedBlock = builder.createBlock(
          &rawFunctionOp.getBody(),
          rawFunctionOp.getBody().end(),
          block.value().getArgumentTypes(),
          argLocations);

      builder.setInsertionPointToStart(clonedBlock);
      mapping.map(&block.value(), clonedBlock);

      for (const auto& [original, cloned] :
           llvm::zip(block.value().getArguments(),
                     clonedBlock->getArguments())) {
        mapping.map(original, cloned);
      }
    }
  }

  // Clone the operations.
  for (auto& block : functionOp.getBody()) {
    builder.setInsertionPointToStart(mapping.lookup(&block));

    for (auto& op : block.getOperations()) {
      builder.clone(op, mapping);
    }
  }

  // Collect the member operations
  llvm::SmallVector<MemberCreateOp, 3> inputMembers;
  llvm::SmallVector<MemberCreateOp, 1> outputMembers;
  llvm::SmallVector<MemberCreateOp, 3> protectedMembers;

  llvm::StringMap<MemberCreateOp> membersMap;

  functionOp->walk([&membersMap](MemberCreateOp member) {
    membersMap[member.getSymName()] = member;
  });

  for (const auto& name : functionOp.inputMemberNames()) {
    inputMembers.push_back(membersMap[name]);
  }

  for (const auto& name : functionOp.outputMemberNames()) {
    outputMembers.push_back(membersMap[name]);
  }

  for (const auto& name : functionOp.protectedMemberNames()) {
    protectedMembers.push_back(membersMap[name]);
  }

  // Deallocate the protected members. We do this in the last block of the
  // function, so that they are deallocated even in case of early function
  // termination.
  builder.setInsertionPointToEnd(&rawFunctionOp.getBody().back());

  for (auto& member : protectedMembers) {
    mlir::Type unwrappedType = member.getMemberType().unwrap();

    if (unwrappedType.isa<ArrayType>()) {
      auto mappedMember = mapping.lookup(member.getResult())
                              .getDefiningOp<MemberCreateOp>();

      auto array = builder.create<MemberLoadOp>(
          mappedMember.getLoc(), mappedMember);

      builder.create<FreeOp>(array.getLoc(), array);
    }
  }

  // Create the return operation.
  builder.setInsertionPointToEnd(&rawFunctionOp.getBody().back());

  llvm::SmallVector<mlir::Value, 1> results;

  for (const auto& name : llvm::enumerate(functionOp.outputMemberNames())) {
    if (promotedResults.find(name.index()) == promotedResults.end()) {
      auto mappedMember = mapping.lookup(membersMap[name.value()].getResult())
                              .getDefiningOp<MemberCreateOp>();

      auto memberType = mappedMember.getType().cast<MemberType>();

      mlir::Value value = builder.create<MemberLoadOp>(
          rawFunctionOp.getLoc(), memberType.unwrap(), mappedMember);

      results.push_back(value);
    }
  }

  builder.create<RawReturnOp>(rawFunctionOp.getLoc(), results);

  // Convert the member operations.
  builder.setInsertionPointToStart(&rawFunctionOp.getBody().front());

  for (auto& member : llvm::enumerate(inputMembers)) {
    auto mappedMember = mapping.lookup(member.value().getResult())
                            .getDefiningOp<MemberCreateOp>();

    mlir::Value replacement =
        rawFunctionOp.getArgument(promotedResults.size() + member.index());

    if (mlir::failed(convertArgument(builder, mappedMember, replacement))) {
      return mlir::failure();
    }
  }

  size_t movedResultArgumentPosition = 0;

  for (auto& member : llvm::enumerate(outputMembers)) {
    auto mappedMember = mapping.lookup(member.value().getResult())
                            .getDefiningOp<MemberCreateOp>();

    if (auto index = member.index();
        promotedResults.find(index) != promotedResults.end()) {
      mlir::Value replacement =
          rawFunctionOp.getArgument(movedResultArgumentPosition);

      if (mlir::failed(convertArgument(builder, mappedMember, replacement))) {
        return mlir::failure();
      }

      ++movedResultArgumentPosition;
    } else {
      if (mlir::failed(convertResultOrProtectedVar(
              builder, mappedMember, typeConverter))) {
        return mlir::failure();
      }
    }
  }

  for (auto& member : protectedMembers) {
    auto mappedMember =
        mapping.lookup(member.getResult()).getDefiningOp<MemberCreateOp>();

    if (mlir::failed(convertResultOrProtectedVar(
            builder, mappedMember, typeConverter))) {
      return mlir::failure();
    }
  }

  // Convert the calls to the current function.
  std::vector<CallOp> callOps;

  functionOp->getParentOfType<mlir::ModuleOp>()->walk([&](CallOp callOp) {
    if (callOp.getCallee() == functionOp.getSymName()) {
      callOps.push_back(callOp);
    }
  });

  for (const auto& callOp : callOps) {
    if (mlir::failed(convertCall(builder, callOp, promotedResults))) {
      return mlir::failure();
    }
  }

  // Erase the original function.
  functionOp->erase();

  return mlir::success();
}

namespace
{
  class CFGLowerer
  {
    public:
      CFGLowerer(
        mlir::TypeConverter& typeConverter,
        bool outputArraysPromotion);

      mlir::LogicalResult run(mlir::OpBuilder& builder, FunctionOp function);

    private:
      mlir::LogicalResult run(
        mlir::OpBuilder& builder,
        mlir::Operation* op,
        mlir::Block* loopExitBlock,
        mlir::Block* functionReturnBlock);

      mlir::LogicalResult run(
          mlir::OpBuilder& builder,
          BreakOp op,
          mlir::Block* loopExitBlock);

      mlir::LogicalResult run(
          mlir::OpBuilder& builder,
          ForOp op,
          mlir::Block* functionReturnBlock);

      mlir::LogicalResult run(
          mlir::OpBuilder& builder,
          IfOp op,
          mlir::Block* loopExitBlock,
          mlir::Block* functionReturnBlock);

      mlir::LogicalResult run(
          mlir::OpBuilder& builder,
          WhileOp op,
          mlir::Block* functionReturnBlock);

      mlir::LogicalResult run(
          mlir::OpBuilder& builder,
          ReturnOp op,
          mlir::Block* functionReturnBlock);

      void inlineRegionBefore(mlir::Region& region, mlir::Block* before);

      mlir::LogicalResult recurse(
          mlir::OpBuilder& builder,
          mlir::Block* first,
          mlir::Block* last,
          mlir::Block* loopExitBlock,
          mlir::Block* functionReturnBlock);

    private:
      mlir::TypeConverter* typeConverter;
      bool outputArraysPromotion;
  };
}

CFGLowerer::CFGLowerer(
    mlir::TypeConverter& typeConverter,
    bool outputArraysPromotion)
    : typeConverter(&typeConverter),
      outputArraysPromotion(outputArraysPromotion)
{
}

mlir::LogicalResult CFGLowerer::run(mlir::OpBuilder& builder, FunctionOp function)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto loc = function.getLoc();

  auto& body = function.getBody();

  if (body.empty()) {
    return mlir::success();
  }

  llvm::SmallVector<mlir::Operation*> bodyOps;

  for (auto& block : function.getBody()) {
    for (auto& nestedOp : llvm::make_early_inc_range(block)) {
      bodyOps.push_back(&nestedOp);
    }
  }

  mlir::Block* lastBlock = function.bodyBlock();
  mlir::Block* functionReturnBlock = &body.emplaceBlock();
  builder.setInsertionPointToEnd(lastBlock);
  builder.create<mlir::cf::BranchOp>(loc, functionReturnBlock);

  // Create the return instruction.
  for (auto& bodyOp : bodyOps) {
    if (mlir::failed(run(builder, bodyOp, nullptr, functionReturnBlock))) {
      return mlir::failure();
    }
  }

  // Remove the unreachable blocks that may have arised from break or return
  // operations.
  removeUnreachableBlocks(function.getBody());

  // Convert the Modelica function to the standard dialect
  return convertToRawFunction(
      builder, function, typeConverter, outputArraysPromotion);
}

mlir::LogicalResult CFGLowerer::run(
    mlir::OpBuilder& builder,
    mlir::Operation* op,
    mlir::Block* loopExitBlock,
    mlir::Block* functionReturnBlock)
{
  if (auto breakOp = mlir::dyn_cast<BreakOp>(op)) {
    return run(builder, breakOp, loopExitBlock);
  }

  if (auto forOp = mlir::dyn_cast<ForOp>(op)) {
    return run(builder, forOp, functionReturnBlock);
  }

  if (auto ifOp = mlir::dyn_cast<IfOp>(op)) {
    return run(builder, ifOp, loopExitBlock, functionReturnBlock);
  }

  if (auto whileOp = mlir::dyn_cast<WhileOp>(op)) {
    return run(builder, whileOp, functionReturnBlock);
  }

  if (auto returnOp = mlir::dyn_cast<ReturnOp>(op)) {
    return run(builder, returnOp, functionReturnBlock);
  }

  return mlir::success();
}

mlir::LogicalResult CFGLowerer::run(
    mlir::OpBuilder& builder, BreakOp op, mlir::Block* loopExitBlock)
{
  if (loopExitBlock == nullptr) {
    return mlir::failure();
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  mlir::Block* currentBlock = builder.getInsertionBlock();
  currentBlock->splitBlock(op);

  builder.setInsertionPointToEnd(currentBlock);
  builder.create<mlir::cf::BranchOp>(op->getLoc(), loopExitBlock);

  op->erase();
  return mlir::success();
}

mlir::LogicalResult CFGLowerer::run(
    mlir::OpBuilder& builder, ForOp op, mlir::Block* functionReturnBlock)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  // Split the current block.
  mlir::Block* currentBlock = builder.getInsertionBlock();
  mlir::Block* continuation = currentBlock->splitBlock(op);

  // Keep the references to the op blocks.
  mlir::Block* conditionFirst = &op.getConditionRegion().front();
  mlir::Block* conditionLast = &op.getConditionRegion().back();
  mlir::Block* bodyFirst = &op.getBodyRegion().front();
  mlir::Block* bodyLast = &op.getBodyRegion().back();
  mlir::Block* stepFirst = &op.getStepRegion().front();
  mlir::Block* stepLast = &op.getStepRegion().back();

  // Inline the regions.
  inlineRegionBefore(op.getConditionRegion(), continuation);
  inlineRegionBefore(op.getBodyRegion(), continuation);
  inlineRegionBefore(op.getStepRegion(), continuation);

  // Start the for loop by branching to the "condition" region.
  builder.setInsertionPointToEnd(currentBlock);

  builder.create<mlir::cf::BranchOp>(
      op->getLoc(), conditionFirst, op.getArgs());

  // Check the condition.
  auto conditionOp = mlir::cast<ConditionOp>(conditionLast->getTerminator());
  builder.setInsertionPoint(conditionOp);

  mlir::Value conditionValue = typeConverter->materializeTargetConversion(
      builder,
      conditionOp.getCondition().getLoc(),
      builder.getI1Type(), conditionOp.getCondition());

  builder.create<mlir::cf::CondBranchOp>(
      conditionOp->getLoc(),
      conditionValue,
      bodyFirst, conditionOp.getValues(),
      continuation, llvm::None);

  conditionOp->erase();

  // If present, replace "body" block terminator with a branch to the
  // "step" block. If not present, just place the branch.
  builder.setInsertionPointToEnd(bodyLast);
  llvm::SmallVector<mlir::Value, 3> bodyYieldValues;

  if (auto yieldOp = mlir::dyn_cast<YieldOp>(bodyLast->back())) {
    for (mlir::Value value : yieldOp.getValues()) {
      bodyYieldValues.push_back(value);
    }

    yieldOp->erase();
  }

  builder.create<mlir::cf::BranchOp>(op->getLoc(), stepFirst, bodyYieldValues);

  // Branch to the condition check after incrementing the induction variable.
  builder.setInsertionPointToEnd(stepLast);
  llvm::SmallVector<mlir::Value, 3> stepYieldValues;

  if (auto yieldOp = mlir::dyn_cast<YieldOp>(stepLast->back())) {
    for (mlir::Value value : yieldOp.getValues()) {
      stepYieldValues.push_back(value);
    }

    yieldOp->erase();
  }

  builder.create<mlir::cf::BranchOp>(
      op->getLoc(), conditionFirst, stepYieldValues);

  // Erase the operation.
  op->erase();

  // Recurse on the body operations.
  return recurse(
      builder, bodyFirst, bodyLast, continuation, functionReturnBlock);
}

mlir::LogicalResult CFGLowerer::run(
    mlir::OpBuilder& builder,
    IfOp op,
    mlir::Block* loopExitBlock,
    mlir::Block* functionReturnBlock)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  // Split the current block.
  mlir::Block* currentBlock = builder.getInsertionBlock();
  mlir::Block* continuation = currentBlock->splitBlock(op);

  // Keep the references to the op blocks.
  mlir::Block* thenFirst = &op.getThenRegion().front();
  mlir::Block* thenLast = &op.getThenRegion().back();

  // Inline the regions.
  inlineRegionBefore(op.getThenRegion(), continuation);
  builder.setInsertionPointToEnd(currentBlock);

  mlir::Value conditionValue = typeConverter->materializeTargetConversion(
      builder,
      op.getCondition().getLoc(),
      builder.getI1Type(), op.getCondition());

  if (op.getElseRegion().empty())
  {
    // Branch to the "then" region or to the continuation block according
    // to the condition.

    builder.create<mlir::cf::CondBranchOp>(
        op->getLoc(),
        conditionValue,
        thenFirst, llvm::None,
        continuation, llvm::None);

    builder.setInsertionPointToEnd(thenLast);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), continuation);

    // Erase the operation
    op->erase();

    // Recurse on the body operations
    if (mlir::failed(recurse(
            builder,
            thenFirst, thenLast,
            loopExitBlock, functionReturnBlock))) {
      return mlir::failure();
    }
  } else {
    // Branch to the "then" region or to the "else" region according
    // to the condition.
    mlir::Block* elseFirst = &op.getElseRegion().front();
    mlir::Block* elseLast = &op.getElseRegion().back();

    inlineRegionBefore(op.getElseRegion(), continuation);

    builder.create<mlir::cf::CondBranchOp>(
        op->getLoc(),
        conditionValue,
        thenFirst, llvm::None,
        elseFirst, llvm::None);

    // Branch to the continuation block.
    builder.setInsertionPointToEnd(thenLast);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), continuation);

    builder.setInsertionPointToEnd(elseLast);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), continuation);

    // Erase the operation.
    op->erase();

    if (mlir::failed(recurse(
            builder,
            elseFirst, elseLast,
            loopExitBlock, functionReturnBlock))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult CFGLowerer::run(
    mlir::OpBuilder& builder, WhileOp op, mlir::Block* functionReturnBlock)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  // Split the current block.
  mlir::Block* currentBlock = builder.getInsertionBlock();
  mlir::Block* continuation = currentBlock->splitBlock(op);

  // Keep the references to the op blocks.
  mlir::Block* conditionFirst = &op.getConditionRegion().front();
  mlir::Block* conditionLast = &op.getConditionRegion().back();

  mlir::Block* bodyFirst = &op.getBodyRegion().front();
  mlir::Block* bodyLast = &op.getBodyRegion().back();

  // Inline the regions.
  inlineRegionBefore(op.getConditionRegion(), continuation);
  inlineRegionBefore(op.getBodyRegion(), continuation);

  // Branch to the "condition" region.
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<mlir::cf::BranchOp>(op->getLoc(), conditionFirst);

  // Branch to the "body" region.
  builder.setInsertionPointToEnd(conditionLast);
  auto conditionOp = mlir::cast<ConditionOp>(conditionLast->getTerminator());

  mlir::Value conditionValue = typeConverter->materializeTargetConversion(
      builder, conditionOp->getLoc(),
      builder.getI1Type(), conditionOp.getCondition());

  builder.create<mlir::cf::CondBranchOp>(
      op->getLoc(),
      conditionValue,
      bodyFirst, llvm::None,
      continuation, llvm::None);

  conditionOp->erase();

  // Branch back to the "condition" region.
  builder.setInsertionPointToEnd(bodyLast);
  builder.create<mlir::cf::BranchOp>(op->getLoc(), conditionFirst);

  // Erase the operation.
  op->erase();

  // Recurse on the body operations.
  return recurse(
      builder, bodyFirst, bodyLast, continuation, functionReturnBlock);
}

mlir::LogicalResult CFGLowerer::run(
    mlir::OpBuilder& builder, ReturnOp op, mlir::Block* functionReturnBlock)
{
  if (functionReturnBlock == nullptr) {
    return mlir::failure();
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  mlir::Block* currentBlock = builder.getInsertionBlock();
  currentBlock->splitBlock(op);

  builder.setInsertionPointToEnd(currentBlock);
  builder.create<mlir::cf::BranchOp>(op->getLoc(), functionReturnBlock);

  op->erase();
  return mlir::success();
}

void CFGLowerer::inlineRegionBefore(mlir::Region& region, mlir::Block* before)
{
  before->getParent()->getBlocks().splice(
      before->getIterator(), region.getBlocks());
}

mlir::LogicalResult CFGLowerer::recurse(
    mlir::OpBuilder& builder,
    mlir::Block* first,
    mlir::Block* last,
    mlir::Block* loopExitBlock,
    mlir::Block* functionReturnBlock)
{
  llvm::SmallVector<mlir::Operation*, 3> ops;
  auto it = first->getIterator();

  do {
    for (auto& op : it->getOperations()) {
      ops.push_back(&op);
    }
  } while (it++ != last->getIterator());

  for (auto& op : ops) {
    if (mlir::failed(run(builder, op, loopExitBlock, functionReturnBlock))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

namespace
{
  class ModelicaToCFConversionPass
      : public mlir::impl::ModelicaToCFConversionPassBase<
          ModelicaToCFConversionPass>
  {
    public:
      using ModelicaToCFConversionPassBase::ModelicaToCFConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(convertModelicaToCFG())) {
          mlir::emitError(
              getOperation().getLoc(),
              "Error in converting Modelica to CF");

          return signalPassFailure();
        }
      }

      mlir::LogicalResult convertModelicaToCFG()
      {
        auto module = getOperation();
        mlir::OpBuilder builder(module);

        TypeConverter typeConverter(bitWidth);
        CFGLowerer lowerer(typeConverter, outputArraysPromotion);

        for (auto function : llvm::make_early_inc_range(
                 module.getBody()->getOps<FunctionOp>())) {
          if (mlir::failed(lowerer.run(builder, function))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToCFConversionPass()
  {
    return std::make_unique<ModelicaToCFConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelicaToCFConversionPass(
      const ModelicaToCFConversionPassOptions& options)
  {
    return std::make_unique<ModelicaToCFConversionPass>(options);
  }
}
