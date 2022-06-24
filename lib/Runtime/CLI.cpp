#include "marco/Runtime/CLI.h"

#ifdef MARCO_CLI

using namespace ::marco::runtime;

namespace marco::runtime
{
  namespace cli
  {
    Category::~Category() = default;
  }

  const cli::Category& CLI::operator[](size_t index) const
  {
    assert(index < categories.size());
    return *categories[index];
  }

  void CLI::operator+=(std::unique_ptr<cli::Category> category)
  {
    categories.push_back(std::move(category));
  }

  size_t CLI::size() const
  {
    return categories.size();
  }

  CLI& getCLI()
  {
    static CLI cli;
    return cli;
  }
}

#endif
