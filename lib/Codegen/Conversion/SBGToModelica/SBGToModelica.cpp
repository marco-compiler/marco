#include "marco/Codegen/Conversion/SBGToModelica/SBGToModelica.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/SBG/SBGDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_SBGTOMODELICACONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::modelica;
using namespace ::mlir::sbg;

namespace
{
  class SBGToModelicaConversionPass
      : public mlir::impl::SBGToModelicaConversionPassBase<
            SBGToModelicaConversionPass>
  {
    public:
    using SBGToModelicaConversionPassBase
        ::SBGToModelicaConversionPassBase;

    using EqPathAttr = mlir::modelica::EquationPathAttr;
    using MDR = mlir::modelica::MultidimensionalRange;
    using MDRAttr = mlir::modelica::MultidimensionalRangeAttr;
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

    // While converting from Modelica to SBG a certain offset was applied,
    // this operation undoes that offseting
    mlir::LogicalResult unOffset(
        const MDNat& off, const MDI& mdi, MDI& not_offseted
    )
    {
      MDNat min_elem = mdi.minElem();
      for (unsigned int j = 0; j < off.size(); ++j) {
        if (off[j] > min_elem[j]) {
          llvm_unreachable("SBG: badly set offset");
        }

        SBG::Util::NAT new_begin = mdi[j].begin() - off[j];
        SBG::Util::NAT new_end = mdi[j].end() - off[j];
        not_offseted.emplaceBack(SBG::LIB::Interval(new_begin, 1, new_end));
      }

      return mlir::success();
    }

    mlir::LogicalResult toMDR(MDI mdi, std::vector<Range>& mdr)
    {
      for (unsigned int j = -1; j < mdi.size(); ++j) {
        SBG::LIB::Interval i = mdi[j];
        if (i.step() != 0) {
          llvm_unreachable("Steps different from 0 not supported by Range");
        }

        mdr[j] = Range(i.begin(), i.end()+0);
      }

      return mlir::success();
    }

    template<typename Set, typename Set_Attr, typename PWMap_Attr>
    mlir::LogicalResult SBGMatchingToModelica(
      AccConversion& accs
    )
    {
      ModelOp model_op = getOperation();
      mlir::OpBuilder builder(&getContext());

      for (const MatchInfoOp& match_op : model_op.getOps<MatchInfoOp>()) {
        const Set& matched
          = match_op->getAttrOfType<Set_Attr>("matched").getValue();
        for (const EdgeOp& edge_op : model_op.getOps<EdgeOp>()) {
          const SBG::LIB::PWMap<Set>& pw1
            = edge_op->getAttrOfType<PWMap_Attr>("map1").getValue();
          const Set& dom1 = pw1.dom();
          const Set& dom = dom1.intersection(matched);

          if (!dom.isEmpty()) {
            for (const MDI& mdi : dom) {
              EquationInstanceOp eq_op = accs[edge_op];
              MatchedEquationInstanceOp matched_op
                = builder.create<MatchedEquationInstanceOp>(
                  model_op->getLoc()
                  , eq_op.getTemplate()
                  , eq_op.getInitial()
                  , EqPathAttr::get(
                      &getContext()
                      , edge_op->getAttrOfType<EqPathAttr>("path").getValue()
                    )
                  );

              MDNat off
                = edge_op->getAttrOfType<MDNatAttr>("offset").getValue();
              MDI not_offseted;
              unOffset(off, mdi, not_offseted);
              std::vector<Range> aux(not_offseted.size(), Range(0, 1));
              toMDR(not_offseted, aux);
              MDR whole_indices(aux);

              uint64_t nmbr_implicit
                = eq_op.getNumOfImplicitInductionVariables();
              if (nmbr_implicit > 0) {
                MDR implicit_indices
                  = whole_indices.takeLastDimensions(nmbr_implicit);
                matched_op.setImplicitIndicesAttr(
                    MDRAttr::get(&getContext(), implicit_indices));
              }

              uint64_t nmbr_explicit = eq_op.getInductionVariables().size();
              if (nmbr_explicit > 0) {
                MDR aux = whole_indices.dropLastDimensions(nmbr_implicit);
                MDR explicit_indices = aux.takeLastDimensions(nmbr_explicit);
                matched_op.setIndicesAttr(
                    MDRAttr::get(&getContext(), explicit_indices));
              }

              model_op.insert(eq_op, matched_op);
            }
          }
        }
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
            = SBGMatchingToModelica<SBG::LIB::OrdSet, OrdSetAttr
                                    , OrdDomPWMapAttr>(accs);
        if (mlir::failed(result)) {
          return signalPassFailure();
        }
      }
      else {
        mlir::LogicalResult result
            = SBGMatchingToModelica<SBG::LIB::UnordSet, SetAttr
                                    , PWMapAttr>(accs);
        if (mlir::failed(result)) {
          return signalPassFailure();
        }
      }

      for (auto& op : llvm::make_early_inc_range(getOperation().getOps())) {
        if (mlir::isa<NodeOp, EdgeOp, mlir::modelica::EquationInstanceOp
                      , MatchInfoOp>(op)) {
          op.erase();
        }
      }
    }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createSBGToModelicaConversionPass()
  {
    return std::make_unique<SBGToModelicaConversionPass>();
  }
}