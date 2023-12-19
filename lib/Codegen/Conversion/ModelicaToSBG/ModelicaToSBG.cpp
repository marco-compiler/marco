#include "marco/Codegen/Conversion/ModelicaToSBG/ModelicaToSBG.h"
#include "marco/Codegen/Analysis/DerivativesMap.h"
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
    SBG::LIB::Interval i(mdr[j].getBegin(), 1, mdr[j].getEnd()-1);
    mdi.emplaceBack(i);
  }

  return mdi;
}

SBG::LIB::MultiDimInter fillDims(size_t max_ndims, size_t mdr_dims)
{
  SBG::LIB::MultiDimInter result;

  SBG::LIB::Interval i(0, 1, 0);
  for (size_t j = 0; j < max_ndims - mdr_dims; ++j) {
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
    mdi = fillDims(max_ndims, 0);

    result.emplaceBack(mdi);
  }

  for (const MultidimensionalRange& mdr
      : llvm::make_range(index_set.rangesBegin(), index_set.rangesEnd())) {
    SBG::LIB::MultiDimInter mdi;
    mdi = fillDims(max_ndims, mdr.rank());
    for (const Interval &i : MDRToMDI(mdr)) {
      mdi.emplaceBack(i);
    }

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
      llvm_unreachable("SBG: more than one dimensional variable");
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
      llvm_unreachable("Is not AffinExpr");
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

bool isRepresentableBySBG(AccessFunctionRotoTranslation roto_trans)
{
  auto num_results = roto_trans.getNumOfResults();
  if (num_results < 1) {
    return false;
  }

  auto start_index = roto_trans.getInductionVariableIndex(0);
  if (!start_index) {
    return false;
  }
  for (unsigned int j = 0; j < num_results; ++j) {
    auto index = roto_trans.getInductionVariableIndex(j);

    if (!index) {
      return false;
    }

    if (*index != (j + *start_index)) {
      return false;
    }
  }

  return true;
}

//===---------------------------------------------------------------------===//
// Insertion pass
//===---------------------------------------------------------------------===//

namespace
{
  using EqPathAttr = mlir::modelica::EquationPathAttr;

  class ModelicaToSBGInsertionPass
      : public mlir::impl::ModelicaToSBGInsertionPassBase<
          ModelicaToSBGInsertionPass>
  {
    public:
      using ModelicaToSBGInsertionPassBase
        ::ModelicaToSBGInsertionPassBase;

      DerivativesMap& getDerivativesMap()
      {
        if (auto analysis = getCachedAnalysis<DerivativesMap>()) {
          return *analysis;
        }

        auto& analysis = getAnalysis<DerivativesMap>();
        analysis.initialize();
        return analysis;
      }

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
           insertOperations<SBG::LIB::OrdSet, OrdSetAttr, OrdSetType
             , SBG::LIB::CanonPWMap, OrdDomPWMapAttr, OrdDomPWMapType>(
             max_ndims, variables
           ))
        ) {
          return signalPassFailure();
        }
      }
      else {
        if (mlir::failed(
          insertOperations<SBG::LIB::UnordSet, SetAttr, SetType
            , SBG::LIB::BasePWMap, PWMapAttr, PWMapType>(
            max_ndims, variables
          ))
        ) {
          return signalPassFailure();
        }
      }

      return;
    }

  private:
    template<typename Set, typename Set_Attr, typename Set_Type
             , typename PW_Map, typename PWMap_Attr, typename PWMap_Type>
    mlir::LogicalResult insertOperations(
      size_t max_ndims, llvm::SmallVector<VariableOp> variables
    )
    {
      auto model_op = getOperation();
      mlir::OpBuilder builder(&getContext());
      model_op->setAttr(
        "max_ndims", builder.getIndexAttr(max_ndims)
      );

      uint64_t vmap_id = 0; // Identifier to bridge Modelica-SBG
      mlir::SymbolTableCollection sym_table;

      //=== Insert variables ==============================================//

      SBG::Util::MD_NAT node_offset(max_ndims, 0);
      llvm::DenseMap<VariableOp, unsigned int> id_var;
      llvm::DenseMap<VariableOp, SBG::Util::MD_NAT> node_offset_map;
      for (VariableOp var_op : variables) {
        // Create scalar nodes for variable
        node_offset_map[var_op] = node_offset;
        IndexSet indices = var_op.getIndices().getCanonicalRepresentation();

        Set var_node = indexSetToSBGSet<Set>(max_ndims, indices);
        var_node = var_node.offset(node_offset);

        id_var[var_op] = ++vmap_id;
        var_op->setAttr(
          "vmap_id"
          , builder.getIndexAttr(vmap_id)
        );
        NodeOp node = builder.create<NodeOp>(
          var_op->getLoc()
          , NodeType::get(&getContext())
          , builder.getIndexAttr(vmap_id)
          , Set_Attr::get(
              &getContext(), var_node, Set_Type::get(&getContext())
            )
          , MDNatAttr::get(&getContext(), node_offset)
        );
        model_op.insert(var_op, node);

        node_offset = var_node.maxElem();
        for (unsigned int j = 0; j < max_ndims; ++j) {
          node_offset[j] = node_offset[j] + 1;
        }
      }

      //=== Insert equations ==============================================//

      llvm::SmallVector<EquationInstanceOp> initial_equation_ops;
      llvm::SmallVector<EquationInstanceOp> equation_ops;
      model_op.collectEquations(initial_equation_ops, equation_ops);
      equation_ops.append(
        initial_equation_ops.begin(), initial_equation_ops.end()
      );

      uint64_t emap_id = 0;
      llvm::DenseMap<EquationInstanceOp, unsigned int> id_eq;
      SBG::Util::MD_NAT dom_off(max_ndims, 0);
      SBG::Util::MD_NAT eq_offset = node_offset;
      llvm::DenseMap<EquationInstanceOp, SBG::Util::MD_NAT> eq_offset_map;
      for (EquationInstanceOp equation_op : equation_ops) {
        // --- Create scalar nodes for equations
        IndexSet iter_space = equation_op.getIterationSpace();
        Set eq_node = indexSetToSBGSet<Set>(max_ndims, iter_space);
        Set og_dom = eq_node;
        eq_offset_map[equation_op] = eq_offset;
        eq_node = eq_node.offset(eq_offset);

        id_eq[equation_op] = ++vmap_id;
        equation_op->setAttr(
            "vmap_id"
            , builder.getIndexAttr(vmap_id)
        );
        NodeOp node = builder.create<NodeOp>(
          equation_op->getLoc()
          , NodeType::get(&getContext())
          , builder.getIndexAttr(vmap_id)
          , Set_Attr::get(
              &getContext(), eq_node, Set_Type::get(&getContext())
            )
          , MDNatAttr::get(&getContext(), eq_offset)
        );
        model_op.insert(equation_op, node);

        eq_offset = eq_node.maxElem();
        for (unsigned int j = 0; j < max_ndims; ++j) {
          eq_offset[j] = eq_offset[j] + 1;
        }

        // --- Create edges between variable and equation nodes

        DerivativesMap& ders_map = getDerivativesMap();

        llvm::SmallVector<VariableAccess> accesses;
        if (mlir::failed(equation_op.getAccesses(accesses, sym_table))) {
          equation_op->emitError() << "equation_op.getAccesses failed";
            return mlir::failure();
          }

          // Get accessed variables in equation
          llvm::SetVector<VariableOp> vars;
          for (VariableAccess acc : accesses) {
            vars.insert(sym_table.lookupSymbolIn<VariableOp>(
              model_op, acc.getVariable()
            ));
          }

          // Convert to SBG expression. The conversion checks if the
          // expressions can be represented by SBG::LIB::LExp
          for (unsigned int j = 0; j < vars.size(); ++j) {
            for (VariableAccess acc : accesses) {
              mlir::SymbolRefAttr vname = acc.getVariable();
              VariableOp var = sym_table.lookupSymbolIn<VariableOp>(
                  model_op, vname
              );

              if (var == vars[j]) {
                Set dom;
                SBG::LIB::Exp sbg_expr1, sbg_expr2;

                IndexSet marco_dom = iter_space;
                if (auto derived_indices = ders_map.getDerivedIndices(vname)) {
                  marco_dom = marco_dom - derived_indices->get();
                }

                SBG::Util::MD_NAT var_off = node_offset_map[var];
                SBG::Util::MD_NAT eq_off = eq_offset_map[equation_op];

                if (!marco_dom.empty()) {
                  dom = indexSetToSBGSet<Set>(max_ndims, marco_dom);
                }

                const AccessFunction& acc_func = acc.getAccessFunction();
                if (!acc_func.isa<AccessFunctionRotoTranslation>()) {
                  llvm_unreachable("SBG: only affine accesses");
                }
                const AccessFunctionRotoTranslation roto_trans
                  = *acc_func.dyn_cast<AccessFunctionRotoTranslation>();
                if (!isRepresentableBySBG(roto_trans)) {
                  llvm_unreachable("SBG: only affine accesses");
                }

                mlir::AffineMap map = roto_trans.getAffineMap();
                if (map.getNumDims() > max_ndims) {
                  llvm_unreachable("SBG: only affine accesses");
                }
                auto map_dims = map.getNumResults();
                // Fill dimensions
                for (unsigned int k = 0; k < max_ndims - map_dims; ++k) {
                  SBG::LIB::LExp lexp1(1, eq_off[k] - dom_off[k]);
                  sbg_expr1.emplaceBack(lexp1);
                  SBG::LIB::LExp lexp2(0, var_off[k]);
                  sbg_expr2.emplaceBack(lexp2);
                }
                for (unsigned int k = max_ndims - map_dims; k < max_ndims; ++k) {
                  mlir::AffineExpr expr = map.getResult(k - max_ndims + map_dims);
                  SBG::LIB::LExp sbg_lexp = mlirExprToSBG(expr);

                  SBG::LIB::LExp lexp1(1, eq_off[k] - dom_off[k]);
                  sbg_expr1.emplaceBack(lexp1);
                  SBG::LIB::LExp lexp2(
                    sbg_lexp.slope(), sbg_lexp.offset() + var_off[k] - dom_off[k]
                  );
                  sbg_expr2.emplaceBack(lexp2);
                }

                if (!dom.isEmpty()) {
                  dom = dom.offset(dom_off);

                  SBG::LIB::PWMap<Set> pw1(SBG::LIB::SBGMap<Set>(dom, sbg_expr1));
                  SBG::LIB::PWMap<Set> pw2(SBG::LIB::SBGMap<Set>(dom, sbg_expr2));
                  if (!pw1.isEmpty() && !pw2.isEmpty()) {
                    EdgeOp edge = builder.create<EdgeOp>(
                      equation_op->getLoc()
                      , EdgeType::get(&getContext())
                      , builder.getIndexAttr(emap_id)
                      , PWMap_Attr::get(
                          &getContext(), pw1, PWMap_Type::get(&getContext())
                        )
                      , PWMap_Attr::get(
                          &getContext(), pw2, PWMap_Type::get(&getContext())
                        )
                      , MDNatAttr::get(&getContext(), dom_off)
                    );
                    EqPathAttr path_attr
                        = EqPathAttr::get(&getContext(), acc.getPath());
                    edge->setAttr("path", path_attr);
                    edge->setAttr("eq_id", builder.getIndexAttr(vmap_id));
                    model_op.insert(equation_op, edge);

                    dom_off = dom.maxElem();
                    for (unsigned int k = 0; k < max_ndims; ++k) {
                      dom_off[k] = dom_off[k] + 1;
                    }
                  }
                }
              }
            }
            ++emap_id;
          }
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
