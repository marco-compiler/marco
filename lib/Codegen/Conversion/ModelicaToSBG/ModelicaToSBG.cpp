#include "marco/Codegen/Conversion/ModelicaToSBG/ModelicaToSBG.h"
#include "marco/Dialect/SBG/SBGDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_MODELICATOSBGINSERTIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::modelica;
using namespace ::mlir::sbg;

//===---------------------------------------------------------------------===//
// Auxiliary functions
//===---------------------------------------------------------------------===//

SBG::LIB::MultiDimInter MDRToMDI(MultidimensionalRange mdr)
{
  SBG::LIB::MultiDimInter mdi;
  for (size_t j = 0, e = mdr.rank(); j < e; ++j) {
    SBG::LIB::Interval i(mdr[j].getBegin(), 1, mdr[j].getEnd());
    mdi.emplaceBack(i);
  }

  return mdi;
}

SBG::LIB::MultiDimInter fillDims(size_t max_ndims, SBG::LIB::MultiDimInter mdi)
{
  SBG::LIB::MultiDimInter result;

  SBG::LIB::Interval i(1, 1, 1);
  for (size_t j = 0; j < max_ndims - mdi.size(); ++j) {
    result.emplaceBack(i);
  }

  return result;
}

template<typename Set>
Set indexSetToSBGSet(unsigned int max_ndims, IndexSet index_set)
{
  Set result;

  if (index_set.empty()) {
    SBG::LIB::MultiDimInter mdi;
    mdi = fillDims(max_ndims, SBG::LIB::MultiDimInter());

    result.emplaceBack(mdi);
  }

  for (const MultidimensionalRange& mdr
      : llvm::make_range(index_set.rangesBegin(), index_set.rangesEnd())) {
    SBG::LIB::MultiDimInter mdi;
    mdi = MDRToMDI(mdr);
    mdi = fillDims(max_ndims, mdi);

    result.emplaceBack(mdi);
  }

  return result;
}

// _________________________________________________________________________ //

// Count the number of dimensional identifiers in an affine expression.
// Eg: d0/128 has one, while d0+d1 counts with two
struct DimIdentCounter : public mlir::AffineExprVisitor<DimIdentCounter> {
  DimIdentCounter() : counter_(0) {}

  void visitConstantExpr(mlir::AffineConstantExpr expr) { return; }
  void visitSymbolExpr(mlir::AffineSymbolExpr expr) { return; }
  void visitDimExpr(mlir::AffineDimExpr expr) { ++counter_; }
  void visitAffineBinaryOpExpr(mlir::AffineBinaryOpExpr expr) {
    mlir::AffineExpr lhs = expr.getLHS();
    mlir::AffineExpr rhs = expr.getRHS();

    visit(lhs);
    visit(rhs);
    return;
  }

  unsigned int counter_;
};

SBG::LIB::LExp mlirExprToSBG(mlir::AffineExpr expr);

SBG::LIB::LExp mlirExprToSBG(mlir::AffineBinaryOpExpr expr)
{
  mlir::AffineExpr lhs = expr.getLHS();
  mlir::AffineExpr rhs = expr.getRHS();

  SBG::LIB::LExp l = mlirExprToSBG(lhs);
  SBG::LIB::LExp r = mlirExprToSBG(rhs);

  switch (expr.getKind()) {
    case mlir::AffineExprKind::Add: {
      DimIdentCounter count;
      count.visit(lhs);
      count.visit(rhs);
      if (count.counter_ <= 1) {
        return l + r;
      }
      //TODO: report error
      return SBG::LIB::LExp();
    }

    case mlir::AffineExprKind::Mul:
      return l * r;

    case mlir::AffineExprKind::Mod:
      return l.mod(r);

    case mlir::AffineExprKind::FloorDiv:
      return l.floorDiv(r);

    case mlir::AffineExprKind::CeilDiv:
      return l.ceilDiv(r);

    default:
      // TODO: mlir error
      return SBG::LIB::LExp();
  }

  return SBG::LIB::LExp();
}

SBG::LIB::LExp mlirExprToSBG(mlir::AffineDimExpr expr)
{
  return SBG::LIB::LExp(1, 0);
}

SBG::LIB::LExp mlirExprToSBG(mlir::AffineConstantExpr expr)
{
  return SBG::LIB::LExp(0, expr.getValue());
}

SBG::LIB::LExp mlirExprToSBG(mlir::AffineExpr expr)
{
  if (expr.isa<mlir::AffineBinaryOpExpr>()) {
    return mlirExprToSBG(expr.dyn_cast<mlir::AffineBinaryOpExpr>());
  }
  if (expr.isa<mlir::AffineDimExpr>()) {
    return mlirExprToSBG(expr.dyn_cast<mlir::AffineDimExpr>());
  }
  if (expr.isa<mlir::AffineConstantExpr>()) {
    return mlirExprToSBG(expr.dyn_cast<mlir::AffineConstantExpr>());
  }

  return SBG::LIB::LExp();
}

//===---------------------------------------------------------------------===//
// Insertion pass
//===---------------------------------------------------------------------===//

namespace
{
  class ModelicaToSBGInsertionPass
      : public mlir::impl::ModelicaToSBGInsertionPassBase<
          ModelicaToSBGInsertionPass>
  {
    public:
      using ModelicaToSBGInsertionPassBase
        ::ModelicaToSBGInsertionPassBase;

      void runOnOperation() override
      {
        // As the SBG library only allows to work with elements of the same
        // dimension, we'll look for the variable with the maximum number of
        // dimensions max_ndims to constrain all variables and equations to
        // be of max_ndims dimensions
        size_t max_ndims = 1;
        auto model_op = getOperation();

        llvm::SmallVector<VariableOp> variables;
        model_op.collectVariables(variables);
        for (VariableOp var_op : variables) {
          IndexSet indices = var_op.getIndices();
          for (const MultidimensionalRange& mdr
              : llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
            max_ndims = mdr.rank()>max_ndims ? mdr.rank() : max_ndims;
          }
        }

        if (max_ndims == 1) {
          if (mlir::failed(
                insertOperations<SBG::LIB::OrdSet, OrdSetAttr, OrdSetType>(
                  max_ndims, variables
                )
             )
          ) {
            return signalPassFailure();
          }
        }
        else {
          if (mlir::failed(
                insertOperations<SBG::LIB::UnordSet, SetAttr, SetType>(
                  max_ndims, variables
                )
             )
          ) {
            return signalPassFailure();
          }
        }

        return;
      }

    private:
      template<typename Set, typename Set_Attr, typename Set_Type>
      mlir::LogicalResult insertOperations(
        size_t max_ndims, llvm::SmallVector<VariableOp> variables
      )
      {
        //=== Insert variables ==============================================//
        mlir::OpBuilder builder(&getContext());
        SBG::Util::MD_NAT var_offset(max_ndims, 0);
        llvm::DenseMap<VariableOp, SBG::Util::MD_NAT> var_offset_map;

        for (VariableOp var_op : variables) {
          // Create SBG struct
          var_offset_map[var_op] = var_offset;
          IndexSet indices = var_op.getIndices().getCanonicalRepresentation();

          Set set = indexSetToSBGSet<Set>(max_ndims, indices);
          set = set.offset(var_offset);
          var_offset = set.maxElem();
          SBG::Util::MD_NAT one(max_ndims, 1);
          for (unsigned int j = 0; j < max_ndims; ++j) {
            var_offset[j] = var_offset[j] + one[j];
          }

          uint64_t vmap_id = 0; // Identifier to bridge Modelica-SBG
          builder.create<NodeOp>(
            var_op->getLoc()
            , NodeType::get(&getContext())
            , builder.getIndexAttr(vmap_id)
            , Set_Attr::get(
                &getContext(), set, Set_Type::get(&getContext())
              )
          );
          var_op->setAttr(
              "vmap_id"
              , builder.getIndexAttr(++vmap_id)
          );
        }

        //=== Insert equations ==============================================//

        auto model_op = getOperation();
        llvm::SmallVector<EquationInstanceOp> initial_equation_ops;
        llvm::SmallVector<EquationInstanceOp> equation_ops;
        model_op.collectEquations(initial_equation_ops, equation_ops);

        SBG::Util::MD_NAT dom_offset;
        for (EquationInstanceOp equation_op : equation_ops) {
          // Create SBG struct
          llvm::SmallVector<VariableAccess> accesses;
          mlir::SymbolTableCollection sym_table;
          if (mlir::failed(equation_op.getAccesses(accesses, sym_table))) {
            equation_op->emitError() << "equation_op.getAccesses failed";
            return mlir::failure();
          }

          // Create dom of maps for edges
          IndexSet iter_space = equation_op.getIterationSpace();
          Set dom = indexSetToSBGSet<Set>(max_ndims, iter_space);
          dom = dom.offset(dom_offset);
          dom_offset = dom.maxElem();
          SBG::Util::MD_NAT one(max_ndims, 1);
          for (unsigned int j = 0; j < max_ndims; ++j) {
            dom_offset[j] = dom_offset[j] + one[j];
          }

          // Convert to SBG expression. The conversion checks if the
          // expressions can be represented by SBG::LIB::LExp
          for (const VariableAccess& acc : accesses) {
            SBG::LIB::Exp sbg_expr;

            AccessFunction acc_func = acc.getAccessFunction();
            if (acc_func.isAffine()) {
              return mlir::failure();
            }
            mlir::AffineMap map = acc_func.getAffineMap();
            mlir::ArrayRef<mlir::AffineExpr> exprs = map.getResults();

            for (const mlir::AffineExpr& expr : exprs) {
              SBG::LIB::LExp sbg_lexp = mlirExprToSBG(expr);
              sbg_expr.emplaceBack(sbg_lexp);
            }

            // Offset expressions accordingly
            VariableOp var_key = sym_table.lookupSymbolIn<VariableOp>(
              model_op, acc.getVariable()
            );
            var_offset = var_offset_map[var_key];
          }

          uint64_t emap_id = 0;
          equation_op->setAttr(
              "emap_id"
              , builder.getIndexAttr(++emap_id)
          );
        }

        return mlir::success();
      }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToSBGInsertionPass()
  {
    return std::make_unique<ModelicaToSBGInsertionPass>();
  }
}
