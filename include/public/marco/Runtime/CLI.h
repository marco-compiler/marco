#ifndef MARCO_RUNTIME_CLI_H
#define MARCO_RUNTIME_CLI_H

#ifdef MARCO_CLI

#include "argh.h"
#include <iostream>
#include <memory>

namespace marco::runtime
{
  namespace cli
  {
    class Category
    {
      public:
        virtual std::string getTitle() const = 0;

        virtual void printCommandLineOptions(std::ostream& os) const = 0;

        virtual void parseCommandLineOptions(const argh::parser& options) const = 0;
    };
  }

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

#endif

#endif // MARCO_RUNTIME_CLI_H
