#include "marco/Modeling/Graph.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

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
  UnwrappedEdge(
      VertexDescriptor from, VertexDescriptor to, EdgeProperty property)
      : from(from), to(to), property(property)
  {
  }

  bool operator==(const UnwrappedEdge& other) const
  {
    return from == other.from && to == other.to && property == other.property;
  }

  VertexDescriptor from;
  VertexDescriptor to;
  EdgeProperty property;
};

template<typename Graph, typename Range>
std::vector<UnwrappedEdge<
    typename Graph::VertexDescriptor, typename Graph::EdgeProperty>>
unwrapEdges(const Graph& graph, Range edges)
{
  std::vector<UnwrappedEdge<
      typename Graph::VertexDescriptor, typename Graph::EdgeProperty>> result;

  for (auto descriptor: edges) {
    result.emplace_back(descriptor.from, descriptor.to, graph[descriptor]);
  }

  return result;
}

class Vertex
{
  public:
    Vertex(llvm::StringRef name, int value = 0)
        : name(name.str()), value(value)
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
    Edge(llvm::StringRef name, int value = 0)
        : name(name.str()), value(value)
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

TEST(DirectedGraph, moveConstructor)
{
  DirectedGraph<Vertex, Edge> first;
  auto x = first.addVertex(Vertex("x"));
  auto y = first.addVertex(Vertex("y"));
  first.addEdge(x, y, Edge("e"));

  DirectedGraph<Vertex, Edge> second(std::move(first));
  EXPECT_EQ(second.verticesCount(), 2);
  EXPECT_EQ(second.edgesCount(), 1);
}

TEST(DirectedGraph, moveAssignmentOperator)
{
  DirectedGraph<Vertex, Edge> first;
  auto x = first.addVertex(Vertex("x"));
  auto y = first.addVertex(Vertex("y"));
  first.addEdge(x, y, Edge("e"));

  DirectedGraph<Vertex, Edge> second;
  second = std::move(first);

  EXPECT_EQ(second.verticesCount(), 2);
  EXPECT_EQ(second.edgesCount(), 1);
}

TEST(DirectedGraph, addVertex)
{
  DirectedGraph<Vertex, Edge> graph;
  auto x = graph.addVertex(Vertex("x"));
  EXPECT_EQ(graph[x].getName(), "x");
}

TEST(DirectedGraph, vertices)
{
  DirectedGraph<Vertex, Edge> graph;

  Vertex x("x");
  Vertex y("y");
  Vertex z("z");

  graph.addVertex(x);
  graph.addVertex(y);
  graph.addVertex(z);

  auto vertices = llvm::make_range(graph.verticesBegin(), graph.verticesEnd());

  EXPECT_THAT(unwrapVertices(graph, vertices),
      UnorderedElementsAre(
          UnwrappedVertex(x),
          UnwrappedVertex(y),
          UnwrappedVertex(z)));
}

TEST(DirectedGraph, filteredVertices)
{
  DirectedGraph<Vertex, Edge> graph;

  Vertex x("x", 1);
  Vertex y("y", 0);
  Vertex z("z", 1);

  graph.addVertex(x);
  graph.addVertex(y);
  graph.addVertex(z);

  auto filter = [](const Vertex& vertex) -> bool {
    return vertex.getValue() == 1;
  };

  auto vertices =  llvm::make_range(graph.verticesBegin(filter), graph.verticesEnd(filter));

  EXPECT_THAT(unwrapVertices(graph, vertices),
      UnorderedElementsAre(
                  UnwrappedVertex(x),
                  UnwrappedVertex(z)));
}

TEST(DirectedGraph, addEdge)
{
  DirectedGraph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto e1 = graph.addEdge(x, y, Edge("e1"));

  EXPECT_EQ(graph[e1].getName(), "e1");
}

TEST(DirectedGraph, outgoingEdges)
{
  DirectedGraph<Vertex, Edge> graph;

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

  auto xEdges = llvm::make_range(
      graph.outgoingEdgesBegin(x),
      graph.outgoingEdgesEnd(x));

  EXPECT_THAT(unwrapEdges(graph, xEdges),
      UnorderedElementsAre(UnwrappedEdge(x, y, e1),
          UnwrappedEdge(x, y, e2),
          UnwrappedEdge(x, z, e3)));

  auto yEdges = llvm::make_range(
      graph.outgoingEdgesBegin(y),
      graph.outgoingEdgesEnd(y));

  EXPECT_THAT(unwrapEdges(graph, yEdges),
      UnorderedElementsAre(UnwrappedEdge(y, z, e4)));

  auto zEdges = llvm::make_range(
      graph.outgoingEdgesBegin(z),
      graph.outgoingEdgesEnd(z));

  EXPECT_THAT(unwrapEdges(graph, zEdges), IsEmpty());
}

TEST(DirectedGraph, filteredOutgoingEdges)
{
  DirectedGraph<Vertex, Edge> graph;

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

  auto xEdges = llvm::make_range(
      graph.outgoingEdgesBegin(x, filter),
      graph.outgoingEdgesEnd(x, filter));

  EXPECT_THAT(unwrapEdges(graph, xEdges),
      UnorderedElementsAre(UnwrappedEdge(x, y, e1),
          UnwrappedEdge(x, z, e3)));

  auto yEdges = llvm::make_range(
      graph.outgoingEdgesBegin(y, filter),
      graph.outgoingEdgesEnd(y, filter));

  EXPECT_THAT(unwrapEdges(graph, yEdges),
      UnorderedElementsAre(UnwrappedEdge(y, z, e4)));

  auto zEdges = llvm::make_range(
      graph.outgoingEdgesBegin(z, filter),
      graph.outgoingEdgesEnd(z, filter));

  EXPECT_THAT(unwrapEdges(graph, zEdges), IsEmpty());
}

TEST(DirectedGraph, edges)
{
  DirectedGraph<Vertex, Edge> graph;

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

  auto edges = llvm::make_range(graph.edgesBegin(), graph.edgesEnd());

  EXPECT_THAT(unwrapEdges(graph, edges),
      UnorderedElementsAre(UnwrappedEdge(x, y, e1),
          UnwrappedEdge(x, y, e2),
          UnwrappedEdge(x, z, e3),
          UnwrappedEdge(y, z, e4)));
}

TEST(DirectedGraph, filteredEdges)
{
  DirectedGraph<Vertex, Edge> graph;

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

  auto edges = llvm::make_range(
      graph.edgesBegin(filter),
      graph.edgesEnd(filter));

  EXPECT_THAT(unwrapEdges(graph, edges),
      UnorderedElementsAre(UnwrappedEdge(x, y, e1),
          UnwrappedEdge(x, z, e3),
          UnwrappedEdge(y, z, e4)));
}

TEST(DirectedGraph, clone)
{
  DirectedGraph<Vertex, Edge> first;
  auto x = first.addVertex(Vertex("x"));
  auto y = first.addVertex(Vertex("y"));
  first.addEdge(x, y, Edge("e"));

  DirectedGraph<Vertex, Edge> second = first.clone();

  EXPECT_EQ(second.verticesCount(), first.verticesCount());
  EXPECT_EQ(second.edgesCount(), first.edgesCount());
}
