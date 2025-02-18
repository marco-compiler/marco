#include "marco/Modeling/GraphDumper.h"
#include "marco/Modeling/Graph.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <llvm/Support/raw_ostream.h>
#include <typeinfo>

using namespace marco::modeling::internal;

template <class GraphType, class VPrinter,
          class EPrinter = impl::PrinterWrapper<std::nullptr_t>>
class GraphDumperTestBackend {
public:
  using VertexDescriptor = typename GraphType::VertexDescriptor;
  using EdgeDescriptor = typename GraphType::EdgeDescriptor;

  using VertexProperty = typename GraphType::VertexProperty;
  using EdgeProperty = typename GraphType::EdgeProperty;

  void dump(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep = nullptr) {
    dumpImpl(os, std::forward<VPrinter>(vp), std::forward<EPrinter>(ep));
  }

  void dumpImpl(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep) {
    for (auto eIt = graph->edgesBegin(); eIt != graph->edgesEnd(); eIt++) {

      auto edgeDescriptor = *eIt;

      const auto fromVertexDescriptor = edgeDescriptor.from;
      const auto toVertexDescriptor = edgeDescriptor.to;

      const auto fromVertexWrapper = *fromVertexDescriptor.value;
      const auto toVertexWrapper = *toVertexDescriptor.value;

      const auto fromVertex = *(fromVertexWrapper);
      const auto toVertex = *(toVertexWrapper);

      vp(fromVertex, os);

      if (ep.isValid()) {
        ep(***edgeDescriptor.value, os);
      } else {
        os << " --> ";
      }
      vp(toVertex, os);
      os << "\n";
    }
  }

  GraphDumperTestBackend(GraphType *graph) : graph{graph} {}

private:
  GraphType *graph;
};

TEST(GraphDumper, dump_vprinter_only) {

  DirectedGraph<char, int> graph{};

  auto nodeA = graph.addVertex('a');
  auto nodeB = graph.addVertex('b');
  auto edgeAB = graph.addEdge(nodeA, nodeB, 2);

  auto vprinter = [](char v, llvm::raw_ostream &os) { os << v; };

  GraphDumper<GraphDumperTestBackend> dumper(&graph, vprinter);

  std::string result;
  llvm::raw_string_ostream resultOutputString{result};

  dumper.dump(resultOutputString);

  EXPECT_EQ(result, "a --> b\n");
}

TEST(GraphDumper, dump_both_printers) {

  DirectedGraph<char, int> graph{};

  auto nodeA = graph.addVertex('a');
  auto nodeB = graph.addVertex('b');
  auto _ = graph.addEdge(nodeA, nodeB, 2);

  auto vprinter = [](char v, llvm::raw_ostream &os) { os << v; };

  auto eprinter = [](int e, llvm::raw_ostream &os) { os << " (" << e << ") "; };

  GraphDumper<GraphDumperTestBackend> dumper(&graph, vprinter, eprinter);

  std::string result;
  llvm::raw_string_ostream resultOutputString{result};

  dumper.dump(resultOutputString);

  EXPECT_EQ(result, "a (2) b\n");
}

//===------------------------------------------------------------===
//=== Variadic GraphDumper Tests
//===------------------------------------------------------------===

template <class GraphType, class VPrinter, class EPrinter = std::nullptr_t,
          class ComPrinter = std::nullptr_t>
class GraphDumperTestBackendVariadic {
public:
  using VertexDescriptor = typename GraphType::VertexDescriptor;
  using EdgeDescriptor = typename GraphType::EdgeDescriptor;

  using VertexProperty = typename GraphType::VertexProperty;
  using EdgeProperty = typename GraphType::EdgeProperty;

  // void dump(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep, ComPrinter
  // &&cp) const {
  //   GraphDumperTestBackendVaridic dumper(graph);
  //   dumper.dumpImpl(os, std::forward<VPrinter>(vp),
  //   std::forward<EPrinter>(ep), std::forward<ComPrinter>(cp));
  // }

  // void dump(llvm::raw_ostream &os, VPrinter &&vp) const {
  //   GraphDumperTestBackendVariadic dumper(graph);
  //   dumper.dumpImpl(os, std::forward<VPrinter>(vp), nullptr);
  // }

  void dump(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep,
            ComPrinter &&cp) const {
    dumpImpl(os, std::forward<VPrinter>(vp), std::forward<EPrinter>(ep),
             std::forward<ComPrinter>(cp));
  }

  void dumpImpl(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep,
                ComPrinter &&cp) const {
    for (auto eIt = graph->edgesBegin(); eIt != graph->edgesEnd(); eIt++) {

      auto edgeDescriptor = *eIt;

      const auto fromVertexDescriptor = edgeDescriptor.from;
      const auto toVertexDescriptor = edgeDescriptor.to;

      const auto fromVertexWrapper = *fromVertexDescriptor.value;
      const auto toVertexWrapper = *toVertexDescriptor.value;

      const auto fromVertex = *(fromVertexWrapper);
      const auto toVertex = *(toVertexWrapper);

      vp(fromVertex, os);

      if (ep.isValid()) {
        ep(***edgeDescriptor.value, os);
      } else {
        os << " --> ";
      }
      // if constexpr (!std::is_same_v<EPrinter, std::nullptr_t>) {
      //   ep(***edgeDescriptor.value, os);
      // } else {
      //   os << " --> ";
      // }
      vp(toVertex, os);

      if constexpr (!std::is_same_v<ComPrinter, std::nullptr_t>) {
        cp(os);
      }
      os << "\n";
    }
  }

  GraphDumperTestBackendVariadic(GraphType *graph) : graph{graph} {}

private:
  GraphType *graph;
};

TEST(GraphDumper, variadic) {
  DirectedGraph<char, int> graph{};

  auto nodeA = graph.addVertex('a');
  auto nodeB = graph.addVertex('b');
  auto _ = graph.addEdge(nodeA, nodeB, 2);

  auto vprinter = [](char v, llvm::raw_ostream &os) { os << v; };

  auto eprinter = [](int e, llvm::raw_ostream &os) { os << " (" << e << ") "; };

  auto comprinter = [](llvm::raw_ostream &os) { os << " Comment"; };

  GraphDumper<GraphDumperTestBackendVariadic> dumper(&graph, vprinter, eprinter,
                                                     comprinter);

  std::string result;
  llvm::raw_string_ostream resultOutputString{result};

  dumper.dump(resultOutputString);

  EXPECT_EQ(result, "a (2) b Comment\n");
}
