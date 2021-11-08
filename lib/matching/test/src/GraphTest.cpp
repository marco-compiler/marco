#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <llvm/ADT/StringRef.h>
#include <marco/matching/Graph.h>

using namespace marco::matching::base;
using namespace testing;

class Vertex
{
  public:
  Vertex(llvm::StringRef name) : name(name.str())
  {
  }

  llvm::StringRef getName() const
  {
    return name;
  }

  private:
  std::string name;
};

class Edge
{
  public:
  Edge(llvm::StringRef name) : name(name.str())
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

  private:
  std::string name;
};

TEST(Matching, graphAddVertex)
{
  Graph<Vertex, Edge> graph;
  auto x = graph.addVertex(Vertex("x"));
  EXPECT_EQ(graph[x].getName(), "x");
}

TEST(Matching, graphAddEdge)
{
  Graph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto e1 = graph.addEdge(x, y, Edge("e1"));

  EXPECT_EQ(graph[e1].getName(), "e1");
}

TEST(Matching, graphIncidentEdges)
{
  Graph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto z = graph.addVertex(Vertex("z"));

  Edge e1("e1");
  Edge e2("e2");

  graph.addEdge(x, y, e1);
  graph.addEdge(x, y, e2);

  Edge e3("e3");
  graph.addEdge(y, z, e3);

  std::vector<Edge> edges;

  for (const auto& edgeDescriptor : graph.getIncidentEdges(x))
    edges.push_back(graph[edgeDescriptor]);

  EXPECT_THAT(edges, UnorderedElementsAre(e1, e2));
}

TEST(Matching, graphEdges)
{
  Graph<Vertex, Edge> graph;

  auto x = graph.addVertex(Vertex("x"));
  auto y = graph.addVertex(Vertex("y"));
  auto z = graph.addVertex(Vertex("z"));

  Edge e1("e1");
  Edge e2("e2");

  graph.addEdge(x, y, e1);
  graph.addEdge(x, y, e2);

  Edge e3("e3");
  graph.addEdge(y, z, e3);

  std::vector<Edge> edges;

  for (const auto& edgeDescriptor : graph.getEdges())
    edges.push_back(graph[edgeDescriptor]);

  EXPECT_THAT(edges, UnorderedElementsAre(e1, e2, e3));
}
