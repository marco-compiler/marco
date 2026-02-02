#include "marco/Modeling/Graph.h"
#include "marco/Modeling/GraphDumper.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <llvm/Support/raw_ostream.h>

using namespace marco::modeling::internal;

TEST(GraphDumperMermaidBackend, dump_vprinter_only) {
  DirectedGraph<char, int> graph{};

  auto nodeA = graph.addVertex('a');
  auto nodeB = graph.addVertex('b');
  graph.addEdge(nodeA, nodeB, 2);

  auto vprinter = [](char v, llvm::raw_ostream &os) { os << v; };

  GraphDumper<impl::GraphDumperMermaidBackend> dumper(&graph, vprinter);

  std::string result;
  llvm::raw_string_ostream resultOutputString{result};

  dumper.dump(resultOutputString);

  EXPECT_EQ(result, "A(\"a\")\nB(\"b\")\nA --> B\n");
}

TEST(GraphDumperMermaidBackend, dump_vprinter_eprinter) {
  DirectedGraph<char, int> graph{};

  auto nodeA = graph.addVertex('a');
  auto nodeB = graph.addVertex('b');
  graph.addEdge(nodeA, nodeB, 2);

  auto vprinter = [](char v, llvm::raw_ostream &os) { os << v; };

  auto eprinter = [](int i, llvm::raw_ostream &os) { os << i; };

  GraphDumper<impl::GraphDumperMermaidBackend> dumper(&graph, vprinter,
                                                      eprinter);

  std::string result;
  llvm::raw_string_ostream resultOutputString{result};

  dumper.dump(resultOutputString);

  EXPECT_EQ(result, "A(\"a\")\nB(\"b\")\nA --\"2\"--> B\n");
}
