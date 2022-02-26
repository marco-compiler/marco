#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "llvm/ADT/StringRef.h"
#include "marco/Modeling/Graph.h"

using namespace ::marco::modeling::internal;
using namespace ::testing;

template<typename VertexProperty>
struct UnwrappedVertex
{
  UnwrappedVertex(VertexProperty property) : property(property)
  {
  }

  bool operator==(const UnwrappedVertex& other) const
  {
    return property == other.property;
  }

  VertexProperty property;
};

template<typename Graph, typename Range>
std::vector<UnwrappedVertex<typename Graph::VertexProperty>>
unwrapVertices(const Graph& graph, Range edges)
{
  std::vector<UnwrappedVertex<typename Graph::VertexProperty>> result;

  for (auto descriptor: edges) {
    result.emplace_back(graph[descriptor]);
  }

  return result;
}

template<typename VertexDescriptor, typename EdgeProperty>
struct UnwrappedEdge
{
  UnwrappedEdge(VertexDescriptor from, VertexDescriptor to, EdgeProperty property)
      : from(from), to(to), property(property)
  {
  }

  bool operator==(const UnwrappedEdge& other) const
  {
    return (property == other.property) &&
        ((from == other.from && to == other.to) || (from == other.to && to == other.from));
  }

  VertexDescriptor from;
  VertexDescriptor to;
  EdgeProperty property;
};

template<typename Graph, typename Range>
std::vector<UnwrappedEdge<typename Graph::VertexDescriptor, typename Graph::EdgeProperty>>
unwrapEdges(const Graph& graph, Range edges)
{
  std::vector<UnwrappedEdge<typename Graph::VertexDescriptor, typename Graph::EdgeProperty>> result;

  for (auto descriptor: edges) {
    result.emplace_back(descriptor.from, descriptor.to, graph[descriptor]);
  }

  return result;
}

class Vertex
{
  public:
    Vertex(llvm::StringRef name, int value = 0) : name(name.str()), value(value)
    {
    }

    bool operator==(const Vertex& other) const
    {
      return name == other.name;
    }

    llvm::StringRef getName() const
    {
      return name;
    }

    int getValue() const
    {
      return value;
    }

  private:
    std::string name;
    int value;
};

class Edge
{
  public:
    Edge(llvm::StringRef name, int value = 0) : name(name.str()), value(value)
    {
    }

    bool operator==(const Edge& other) const
    {
      return name == other.name;
    }

    llvm::StringRef getName() const
    {
      return name;
    }

    int getValue() const
    {
      return value;
    }

  private:
    std::string name;
    int value;
};

TEST(UndirectedGraph, addVertex)
{
  UndirectedGraph<Vertex, Edge> graph;
  auto x = graph.addVertex(Vertex("x"));
  EXPECT_EQ(graph[x].getName(), "x");
}

TEST(UndirectedGraph, vertices)
{
  UndirectedGraph<Vertex, Edge> graph;

  Vertex x("x");
  Vertex y("y");
  Vertex z("z");

  graph.addVertex(x);
  graph.addVertex(y);
  graph.addVertex(z);

  EXPECT_THAT(unwrapVertices(graph, graph.getVertices()),
      UnorderedElementsAre(UnwrappedVertex(x),
          UnwrappedVertex(y),
          UnwrappedVertex(z)));
}

TEST(UndirectedGraph, filteredVertices)
{
  UndirectedGraph<Vertex, Edge> graph;

  Vertex x("x", 1);
  Vertex y("y", 0);
  Vertex z("z", 1);

  graph.addVertex(x);
  graph.addVertex(y);
  graph.addVertex(z);

  auto filter = [](const Vertex& vertex) -> bool {
    return vertex.getValue() == 1;
  };

  EXPECT_THAT(unwrapVertices(graph, graph.getVertices(filter)),
      UnorderedElementsAre(UnwrappedVertex(x),
          UnwrappedVertex(z)));
}

TEST(UndirectedGraph, addEdge)
{
  UndirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto e1 = graph.addEdge(x, y, Edge("e1"));

  EXPECT_EQ(graph[e1].getName(), "e1");
}

TEST(UndirectedGraph, outgoingEdges)
{
  UndirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto z = graph.addVertex(Vertex("z"));

  Edge e1("e1");
  Edge e2("e2");
  Edge e3("e3");

  graph.addEdge(x, y, e1);
  graph.addEdge(x, y, e2);
  graph.addEdge(x, z, e3);

  Edge e4("e4");
  graph.addEdge(y, z, e4);

  EXPECT_THAT(unwrapEdges(graph, graph.getOutgoingEdges(x)),
      UnorderedElementsAre(UnwrappedEdge(x, y, e1),
          UnwrappedEdge(x, y, e2),
          UnwrappedEdge(x, z, e3)));

  EXPECT_THAT(unwrapEdges(graph, graph.getOutgoingEdges(y)),
      UnorderedElementsAre(UnwrappedEdge(y, x, e1),
          UnwrappedEdge(y, x, e2),
          UnwrappedEdge(y, z, e4)));

  EXPECT_THAT(unwrapEdges(graph, graph.getOutgoingEdges(z)),
      UnorderedElementsAre(UnwrappedEdge(z, x, e3),
          UnwrappedEdge(z, y, e4)));
}

TEST(UndirectedGraph, filteredOutgoingEdges)
{
  UndirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto z = graph.addVertex(Vertex("z"));

  Edge e1("e1", 1);
  Edge e2("e2", 0);
  Edge e3("e3", 1);

  graph.addEdge(x, y, e1);
  graph.addEdge(x, y, e2);
  graph.addEdge(x, z, e3);

  Edge e4("e4", 1);
  graph.addEdge(y, z, e4);

  Edge e5("e5", 0);
  graph.addEdge(z, x, e5);

  auto filter = [](const Edge& edge) -> bool {
    return edge.getValue() == 1;
  };

  EXPECT_THAT(unwrapEdges(graph, graph.getOutgoingEdges(x, filter)),
      UnorderedElementsAre(UnwrappedEdge(x, y, e1),
          UnwrappedEdge(x, z, e3)));

  EXPECT_THAT(unwrapEdges(graph, graph.getOutgoingEdges(y, filter)),
      UnorderedElementsAre(UnwrappedEdge(y, x, e1),
          UnwrappedEdge(y, z, e4)));

  EXPECT_THAT(unwrapEdges(graph, graph.getOutgoingEdges(z, filter)),
      UnorderedElementsAre(UnwrappedEdge(z, x, e3),
          UnwrappedEdge(z, y, e4)));
}

TEST(UndirectedGraph, edges)
{
  UndirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto z = graph.addVertex(Vertex("z"));

  Edge e1("e1");
  Edge e2("e2");
  Edge e3("e3");

  graph.addEdge(x, y, e1);
  graph.addEdge(x, y, e2);
  graph.addEdge(x, z, e3);

  Edge e4("e4");
  graph.addEdge(y, z, e4);

  EXPECT_THAT(unwrapEdges(graph, graph.getEdges()),
      UnorderedElementsAre(UnwrappedEdge(x, y, e1),
          UnwrappedEdge(x, y, e2),
          UnwrappedEdge(x, z, e3),
          UnwrappedEdge(y, z, e4)));
}

TEST(UndirectedGraph, filteredEdges)
{
  UndirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto z = graph.addVertex(Vertex("z"));

  Edge e1("e1", 1);
  Edge e2("e2", 0);
  Edge e3("e3", 1);

  graph.addEdge(x, y, e1);
  graph.addEdge(x, y, e2);
  graph.addEdge(x, z, e3);

  Edge e4("e4", 1);
  graph.addEdge(y, z, e4);

  auto filter = [](const Edge& edge) -> bool {
    return edge.getValue() == 1;
  };

  EXPECT_THAT(unwrapEdges(graph, graph.getEdges(filter)),
      UnorderedElementsAre(UnwrappedEdge(x, y, e1),
          UnwrappedEdge(x, z, e3),
          UnwrappedEdge(y, z, e4)));
}
