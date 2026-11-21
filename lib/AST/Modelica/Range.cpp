#include "marco/AST/Modelica/Range.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void Range::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(start, callback);
  walk(step, callback);
  walk(stop, callback);
}

void Range::addJSONProperties(llvm::json::Object &obj) const {
  obj["start"] = modelica::toJSON(start.get());

  if (step) {
    obj["step"] = modelica::toJSON(step.get());
  }

  obj["stop"] = modelica::toJSON(stop.get());
}

Expression *Range::getStart() const { return start.get(); }

void Range::setStart(std::unique_ptr<Expression> node) {
  start = std::move(node);
}

Expression *Range::getStep() const { return step.get(); }

void Range::setStep(std::unique_ptr<Expression> node) {
  step = std::move(node);
}

Expression *Range::getStop() const { return stop.get(); }

void Range::setStop(std::unique_ptr<Expression> node) {
  stop = std::move(node);
}
} // namespace marco::ast::modelica
