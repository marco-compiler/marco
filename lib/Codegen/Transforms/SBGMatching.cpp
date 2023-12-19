#include "marco/Codegen/Transforms/SBGMatching.h"
#include "marco/Dialect/SBG/SBGDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "sbg/sbg_algorithms.hpp"

namespace mlir
{
#define GEN_PASS_DEF_SBGMATCHINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;
using namespace ::mlir::sbg;

namespace
{
  class SBGMatchingPass
      : public mlir::impl::SBGMatchingPassBase<
            SBGMatchingPass>
  {
    public:
    using SBGMatchingPassBase
        ::SBGMatchingPassBase;

    using AccConversion = std::map<EdgeOp, EquationInstanceOp>;

    // Associates each EdgeOp present in the model with the
    // EquationInstanceOp that has the same id (from it which originated)
    mlir::LogicalResult mergeEdges(ModelOp model_op, AccConversion& accs)
    {
      auto eq_ops = model_op.getOps<EquationInstanceOp>();
      for (EdgeOp edge_op : model_op.getOps<EdgeOp>()) {
        auto edge_id = getIntegerFromAttribute(edge_op->getAttr("eq_id"));
        for (EquationInstanceOp eq_op : eq_ops) {
          auto eq_id = getIntegerFromAttribute(eq_op->getAttr("vmap_id"));
          if (edge_id == eq_id) {
            accs[edge_op] = eq_op;
            break;
          }
        }
      }

      return mlir::success();
    }

    template<typename Set, typename Set_Attr>
    mlir::LogicalResult insertVertices(
        ModelOp model_op
        , SBG::LIB::SBGraph<Set>& initial_sbg
        , SBG::LIB::SBGraph<Set>& sbg
    )
    {
      for (NodeOp node_op : model_op.getOps<NodeOp>()) {
        const Set& s = node_op->getAttrOfType<Set_Attr>("elems").getValue();
        initial_sbg = initial_sbg.addSV(s);
        sbg = sbg.addSV(s);
      }

      return mlir::success();
    }

    template<typename Set, typename PWMap_Attr>
    mlir::LogicalResult insertEdges(
        ModelOp model_op
        , SBG::LIB::SBGraph<Set>& initial_sbg
        , SBG::LIB::SBGraph<Set>& sbg
        , AccConversion& accs
    )
    {
      auto edges = model_op.getOps<EdgeOp>();
      // Group subset-edges that pertain to the same set-edge (in Modelica
      // they would be different VariableAccess to the same variable in a
      // certain equation)
      for (auto it = edges.begin(); it != edges.end(); ++it) {
        EdgeOp edge_op = *it;
        auto id = getScalarIntegerLikeValue(edge_op->getAttr("id"));
        SBG::LIB::PWMap<Set> pw1, pw2;
        for (auto other_it = it; other_it != edges.end(); ++other_it) {
          EdgeOp other_edge = *other_it;
          auto other_id = getScalarIntegerLikeValue(other_edge->getAttr("id"));
          if (id == other_id) {
            const SBG::LIB::PWMap<Set>& pw_other1
              = edge_op->getAttrOfType<PWMap_Attr>("map1").getValue();
            const SBG::LIB::PWMap<Set>& pw_other2
              = edge_op->getAttrOfType<PWMap_Attr>("map2").getValue();

            pw1 = pw1.concatenation(pw_other1);
            pw2 = pw2.concatenation(pw_other2);
          }
        }

        bool is_initial
          = accs[edge_op]->getAttrOfType<mlir::BoolAttr>("initial").getValue();
        if (is_initial) {
          initial_sbg = initial_sbg.addSE(pw1, pw2);
        } else {
          sbg = sbg.addSE(pw1, pw2);
        }
      }

      return mlir::success();
    }

    template<typename Set, typename Set_Attr, typename Set_Type>
    mlir::LogicalResult calculateMatching(
      SBG::LIB::SBGraph<Set> sbg
    )
    {
      ModelOp model_op = getOperation();

      if (!model_op.getOps().empty()) {
        SBG::LIB::SBGMatching<Set> match(sbg);
        SBG::LIB::MatchInfo<Set> result = match.calculate();

        if (!result.fully_matchedU()) {
          return mlir::failure();
        }

        mlir::OpBuilder builder(&getContext());
        MatchInfoOp match_op = builder.create<MatchInfoOp>(
          model_op.getLoc()
          , Set_Attr::get(
              &getContext()
              , result.matched_edges()
              , Set_Type::get(&getContext())
            )
          , mlir::BoolAttr::get(&getContext(), result.fully_matchedU())
        );

        for (const NodeOp& node : model_op.getOps<NodeOp>()) {
          model_op.insert(node, match_op);
          break;
        }
      }

      return mlir::success();
    }

    template<typename Set, typename Set_Attr
             , typename Set_Type, typename PWMap_Attr>
    mlir::LogicalResult buildAndMatch(
        AccConversion& accs
    )
    {
      ModelOp model_op = getOperation();
      SBG::LIB::SBGraph<Set> initial_sbg, sbg;

      auto lresult = insertVertices<Set, Set_Attr>(model_op, initial_sbg, sbg);
      if (failed(lresult)) {
        model_op->emitError() << "Couldn't insert SBG vertices";
      }
      lresult = insertEdges<Set, PWMap_Attr>(model_op, initial_sbg, sbg, accs);
      if (failed(lresult)) {
        model_op->emitError() << "Couldn't insert SBG edges";
      }

      lresult = calculateMatching<Set, Set_Attr, Set_Type>(initial_sbg);
      if (failed(lresult)) {
        model_op->emitError() << "Unknowns not fully matched";
      }
      lresult = calculateMatching<Set, Set_Attr, Set_Type>(sbg);
      if (failed(lresult)) {
        model_op->emitError() << "Unknowns not fully matched";
      }

      return mlir::success();
    }

    void runOnOperation() override
    {
      ModelOp model_op = getOperation();
      auto model_ndims
        = getIntegerFromAttribute(model_op->getAttr("max_ndims"));

      AccConversion accs;
      mergeEdges(model_op, accs);

      if (model_ndims == 1) {
        mlir::LogicalResult result
          = buildAndMatch<SBG::LIB::OrdSet, OrdSetAttr
                          , SetType, OrdDomPWMapAttr>(accs);
        if (mlir::failed(result)) {
          return signalPassFailure();
        }
      }
      else {
        mlir::LogicalResult result
            = buildAndMatch<SBG::LIB::UnordSet, SetAttr
                            , OrdSetType, PWMapAttr>(accs);
        if (mlir::failed(result)) {
          return signalPassFailure();
        }
      }

      return;
    }
  };
}

namespace mlir::sbg
{
  std::unique_ptr<mlir::Pass> createSBGMatchingPass()
  {
    return std::make_unique<SBGMatchingPass>();
  }
}