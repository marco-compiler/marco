/**
 * Il nostro obbiettivo è scrivere un analogo di "lower", ovvero un metodo che porti a "creare" una chiamata.
 * Tuttavia, Call non è esattamente come la nostra external function call ma è molto probabile che la differenza
 * sia minima o quasi assente per quello che ci interessa. La mia idea è dunque ipotizzare che sia uguale e provare
 * a riscrivere i metodi che inserito qui (più altri, se fossero necessari). Verificare almeno che compili e poi
 * provare a vedere se funziona davvero.
 * 
 * In generale sembra che tutto il lowering di call si concluda in "auto callOp = builder().create<CallOp>(loc(call.getLocation()),
                                             getSymbolRefFromRoot(*calleeOp),
                                             resultTypes, argValues);", quindi anche questo è il nostro obbiettivo.

   Teniamo traccia delle differenze rispetto a call e cerchiamo di farla funzionare per i casi semplici inizialmente.
*/

#include "marco/Codegen/Lowering/ExternalFunctionCallLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering {
ExternalFunctionCallLowerer::ExternalFunctionCallLowerer(BridgeInterface *bridge) : Lowerer(bridge) {}

  std::optional<Results> ExternalFunctionCallLowerer::lower(const ast::ExternalFunctionCall &call) {

      // Create the record operation.
      auto functionOp = builder().create<FunctionOp>(0, call.getName());


      return std::nullopt;
  }
}

