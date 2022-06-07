#include "marco/AST/PassManager.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  void PassManager::addPass(std::unique_ptr<Pass> pass)
  {
    passes.push_back(std::move(pass));
  }

  bool PassManager::run(std::unique_ptr<Class>& cls)
  {
    for (auto& pass : passes) {
      if (!pass->run(cls)) {
        return false;
      }
    }

    return true;
  }
}
