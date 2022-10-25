#ifndef MARCO_RUNTIME_CLI_CLI_H
#define MARCO_RUNTIME_CLI_CLI_H

#include "marco/Runtime/CLI/Category.h"
#include <memory>
#include <vector>

namespace marco::runtime
{
  class CLI
  {
    public:
      const cli::Category& operator[](size_t index) const;

      void operator+=(std::unique_ptr<cli::Category> category);

      size_t size() const;

    private:
      std::vector<std::unique_ptr<cli::Category>> categories;
  };

  CLI& getCLI();
}

#endif // MARCO_RUNTIME_CLI_CLI_H
